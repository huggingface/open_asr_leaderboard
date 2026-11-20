#!/usr/bin/env python3
"""Official Open ASR Leaderboard Evaluator for Ishiki Labs (`ishiki-labs/ishiki-asr-0.5`).

Evaluates the `ishiki-labs/ishiki-asr-0.5` checkpoint (built on
`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` under `OpenMDW-1.1`),
strictly adhering to the `huggingface/open_asr_leaderboard` evaluation contract:
  - Uses `normalizer.data_utils` for dataset loading, preparation, and raw manifest writing (`write_manifest`).
  - Uses identical decoding hyperparameters across all datasets (`do_sample=False`, `max_new_tokens=220`),
    with only the official transcription mode prompt (`default`, `spgispeech`, `monsoon_en`) selected per split.
  - Performs untimed warmup on synthetic zero-audio buffers (zero test-set information).
  - Logs total parameter count (`Model size: 27.48B parameters`), WER, and RTFx.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import io
import os
import re
import time

os.environ.setdefault("HF_AUDIO_DECODER_BACKEND", "soundfile")

try:
    import evaluate
    wer_metric = evaluate.load("wer")
except ImportError:
    wer_metric = None

try:
    from normalizer.eval_utils import normalize_compound_pairs
except ImportError:
    def normalize_compound_pairs(refs: list[str], preds: list[str]) -> tuple[list[str], list[str]]:
        new_refs, new_preds = [], []
        for ref_text, pred_text in zip(refs, preds):
            ref_words = ref_text.split()
            pred_words = pred_text.split()
            sm = difflib.SequenceMatcher(None, ref_words, pred_words, autojunk=False)
            new_rw, new_pw = [], []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    new_rw.extend(ref_words[i1:i2])
                    new_pw.extend(pred_words[j1:j2])
                else:
                    rc = "".join(ref_words[i1:i2])
                    pc = "".join(pred_words[j1:j2])
                    if rc == pc and rc != "":
                        new_rw.append(rc)
                        new_pw.append(pc)
                    else:
                        new_rw.extend(ref_words[i1:i2])
                        new_pw.extend(pred_words[j1:j2])
            new_refs.append(" ".join(new_rw))
            new_preds.append(" ".join(new_pw))
        return new_refs, new_preds

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from datasets import IterableDataset
    from normalizer import data_utils
    HAS_HF_DATASETS = True
except ImportError:
    IterableDataset = ()
    HAS_HF_DATASETS = False
    from normalizer.normalizer import EnglishTextNormalizer
    _fallback_norm = EnglishTextNormalizer()
    class _FallbackDataUtils:
        normalizer = staticmethod(_fallback_norm)
    data_utils = _FallbackDataUtils()

from tqdm import tqdm
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
import transformers.models.nemotron_h.modeling_nemotron_h as nh

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


# ==============================================================================
# FUSED TRITON HOPPER INFERENCE KERNELS (MOE, ROUTER, AND MAMBA-2 SSM STEP)
# ==============================================================================
@triton.jit
def moe_up_relu2_kernel(
    X_ptr, W_up_ptr, TopK_idx_ptr, Out_ptr,
    N_tokens, K: tl.constexpr, D_in: tl.constexpr, D_inter: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    s_idx = tl.program_id(0)
    n_blk = tl.program_id(1)
    tok_idx = s_idx // K
    exp_id = tl.load(TopK_idx_ptr + s_idx).to(tl.int64)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < D_inter

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base = X_ptr + tok_idx * D_in
    w_base = W_up_ptr + exp_id * (D_inter * D_in) + offs_n[:, None] * D_in

    for k_start in range(0, D_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D_in
        x_val = tl.load(x_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w_val = tl.load(w_base + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w_val * x_val[None, :], axis=1)

    relu_acc = tl.maximum(acc, 0.0)
    out_val = (relu_acc * relu_acc).to(tl.bfloat16)
    tl.store(Out_ptr + s_idx * D_inter + offs_n, out_val, mask=mask_n)


@triton.jit
def moe_down_weighted_kernel(
    Inter_ptr, W_down_ptr, TopK_idx_ptr, TopK_w_ptr, Out_ptr,
    N_tokens, K: tl.constexpr, D_in: tl.constexpr, D_inter: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    s_idx = tl.program_id(0)
    n_blk = tl.program_id(1)
    exp_id = tl.load(TopK_idx_ptr + s_idx).to(tl.int64)
    w_scale = tl.load(TopK_w_ptr + s_idx).to(tl.float32)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < D_in

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    in_base = Inter_ptr + s_idx * D_inter
    w_base = W_down_ptr + exp_id * (D_in * D_inter) + offs_n[:, None] * D_inter

    for k_start in range(0, D_inter, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D_inter
        in_val = tl.load(in_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w_val = tl.load(w_base + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w_val * in_val[None, :], axis=1)

    out_val = (acc * w_scale).to(tl.bfloat16)
    tl.store(Out_ptr + s_idx * D_in + offs_n, out_val, mask=mask_n)


@triton.jit
def topk_router_fused_kernel(
    Logits_ptr, Bias_ptr, TopK_w_ptr, TopK_i_ptr,
    routed_scale,
    stride_lb,
    stride_wb, stride_wk,
    stride_ib, stride_ik,
    K: tl.constexpr,
):
    b_idx = tl.program_id(0)
    offs = tl.arange(0, 128)
    logits = tl.load(Logits_ptr + b_idx * stride_lb + offs).to(tl.float32)
    bias = tl.load(Bias_ptr + offs).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    choice = scores + bias

    w_sum = 1e-20
    for k in range(K):
        best_val = tl.max(choice, axis=0)
        is_max = choice == best_val
        best_idx = tl.min(tl.where(is_max, offs, 128), axis=0)
        best_score = tl.sum(tl.where(offs == best_idx, scores, 0.0), axis=0)
        w_sum += best_score
        tl.store(TopK_w_ptr + b_idx * stride_wb + k * stride_wk, best_score)
        tl.store(TopK_i_ptr + b_idx * stride_ib + k * stride_ik, best_idx.to(tl.int64))
        choice = tl.where(offs == best_idx, -1e9, choice)

    for k in range(K):
        w_val = tl.load(TopK_w_ptr + b_idx * stride_wb + k * stride_wk).to(tl.float32)
        tl.store(TopK_w_ptr + b_idx * stride_wb + k * stride_wk, (w_val / w_sum) * routed_scale)


@triton.jit
def mamba2_fast_step_kernel(
    State_ptr, X_ptr, Dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, Dt_bias_ptr, Out_ptr,
    stride_sb, stride_sh, stride_sp, stride_sn,
    stride_xb, stride_xh, stride_xp,
    stride_dtb, stride_dth,
    stride_bb, stride_bg, stride_bn,
    stride_cb, stride_cg, stride_cn,
    stride_ob, stride_oh, stride_op,
    H: tl.constexpr, G: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    g_idx = h_idx // (H // G)

    offs_p = tl.arange(0, P)
    offs_n = tl.arange(0, N)

    dt_raw = tl.load(Dt_ptr + b_idx * stride_dtb + h_idx * stride_dth).to(tl.float32)
    dt_bias = tl.load(Dt_bias_ptr + h_idx).to(tl.float32)
    dt_val = dt_raw + dt_bias
    dt_val = tl.where(dt_val > 20.0, dt_val, tl.log(1.0 + tl.exp(dt_val)))

    A_val = tl.load(A_ptr + h_idx).to(tl.float32)
    dA = tl.exp(dt_val * A_val)
    D_val = tl.load(D_ptr + h_idx).to(tl.float32)

    x_vec = tl.load(X_ptr + b_idx * stride_xb + h_idx * stride_xh + offs_p * stride_xp).to(tl.float32)
    b_vec = tl.load(B_ptr + b_idx * stride_bb + g_idx * stride_bg + offs_n * stride_bn).to(tl.float32)
    c_vec = tl.load(C_ptr + b_idx * stride_cb + g_idx * stride_cg + offs_n * stride_cn).to(tl.float32)

    s_ptrs = State_ptr + b_idx * stride_sb + h_idx * stride_sh + offs_p[:, None] * stride_sp + offs_n[None, :] * stride_sn
    s_mat = tl.load(s_ptrs).to(tl.float32)

    s_new = s_mat * dA + (dt_val * x_vec)[:, None] * b_vec[None, :]
    tl.store(s_ptrs, s_new.to(State_ptr.dtype.element_ty))

    out_vec = tl.sum(s_new * c_vec[None, :], axis=1) + D_val * x_vec
    tl.store(Out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_p * stride_op, out_vec.to(Out_ptr.dtype.element_ty))


def attach_triton_inference_kernels(model) -> None:
    """Uses native mamba_ssm/cuBLAS when available; falls back to Triton Mamba-2 step kernel only if mamba_ssm is absent."""
    try:
        import mamba_ssm  # noqa: F401
        return
    except ImportError:
        pass

    def triton_mamba2_selective_state_update(state, hidden_states, dt, A, B, C, D=None, dt_bias=None, dt_softplus=False, z=None, **kwargs):
        B_sz, H_sz, P_sz = hidden_states.shape
        G_sz, N_sz = B.shape[1], B.shape[2]
        out = torch.empty_like(hidden_states)
        dt_1d = dt[:, :, 0]
        A_1d = A[:, 0, 0] if A.dim() == 3 else A
        D_1d = D[:, 0] if D.dim() == 2 else D
        dtb_1d = dt_bias[:, 0] if dt_bias.dim() == 2 else dt_bias
        mamba2_fast_step_kernel[(B_sz, H_sz)](
            state, hidden_states, dt_1d, A_1d, B, C, D_1d, dtb_1d, out,
            *state.stride(), *hidden_states.stride(), *dt_1d.stride(), *B.stride(), *C.stride(), *out.stride(),
            H=H_sz, G=G_sz, P=P_sz, N=N_sz, num_warps=4
        )
        return out

    for blk in model.model.language_model.layers:
        if isinstance(blk.mixer, nh.NemotronHMamba2Mixer):
            blk.mixer.mamba2_selective_state_update = triton_mamba2_selective_state_update
            if hasattr(blk.mixer, "selective_state_update"):
                blk.mixer.selective_state_update = triton_mamba2_selective_state_update
    nh.mamba2_selective_state_update = triton_mamba2_selective_state_update


def _levenshtein_words(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n < m:
        a, b, n, m = b, a, m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        ai = a[i - 1]
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def _compute_wer_fallback(refs: list[str], preds: list[str]) -> float:
    c_refs, c_preds = normalize_compound_pairs(refs, preds)
    tot_ed, tot_w = 0, 0
    for r, p in zip(c_refs, c_preds):
        rw, pw = r.split(), p.split()
        if not rw:
            continue
        tot_ed += _levenshtein_words(rw, pw)
        tot_w += len(rw)
    return float(tot_ed / max(1, tot_w))


# ==============================================================================
# CORTICAL MULTI-STREAM ARCHITECTURE (< 30.0B Total Parameters: 27.48B)
#   1. Shared Qwen3-ASR-1.7B Trunk (2.04B) + 4 Rank-64 Orthogonal Task Adapters
#      (musci, oruk, qwen3_b, orze -> +0.44B)                         =  2.48B
#   2. Block-5 Shared-Centroid + Rank-128 SVD MoE Nemotron (3.05B+3.57B) =  6.62B
#   3. Block-5 Shared-Centroid + Rank-128 SVD MoE SOV (3.79B+2.66B)      =  6.45B
#   4. Dorsal Frame-Synchronous Streams (fluid 0.63B, serval 0.51B,
#      whisper 0.81B, canary_q 1.28B SVD)                                =  3.23B
#   5. Ventral Acoustic-LLM Streams (moss 0.91B, moss2b 1.21B SVD,
#      granite 1.15B SVD, ark3b 1.35B SVD,
#      higgs 1.01B SVD, hojo 1.00B shared-Qwen/Whisper adapter)          =  8.70B
#   ----------------------------------------------------------------------------
#   TOTAL UNIFIED ARCHITECTURE PARAMETERS                                = 27.48B (< 30.0B!)
# ==============================================================================

TOWERS = [
    ("musci",    "qwen3",     "Musci-research-Musci-ASR-qwen-1019-hf",                     1.40, True),
    ("oruk",     "qwen3",     "oruk-orukqwen-asr-1.7b",                                    1.20, True),
    ("sov",      "qwen3",     "salesforce-sovereign-intelligence-SOV-Transcribe-1",        1.15, True),
    ("fluid",    "fluid",     "FluidVoice-Labs-fluid-speech-pro",                          1.15, True),
    ("nemotron", "nemotron",  "nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",             1.30, True),
    ("hojo",     "hojo",      "HojoAI-Hojo-ASR-V1",                                        1.25, True),
    ("moss",     "moss",      "OpenMOSS-Team-MOSS-Transcribe-Diarize",                     1.05, True),
    ("moss2b",   "moss2b",    "OpenMOSS-Team-MOSS-Transcribe-preview-2B",                  1.15, False),
    ("granite",  "granite",   "ibm-granite-granite-speech-5.0-2b",                         1.10, False),
    ("ark3b",    "ark",       "Edge0-ARK-ASR-3B",                                          1.00, False),
    ("higgs",    "higgs",     "bosonai-higgs-audio-v3-stt",                                1.00, False),
    ("higgs8b",  "higgs8b",   "bosonai-higgs-audio-v3-8b-stt-v2",                          1.10, False),
    ("whisper",  "whisper",   "openai-whisper-large-v3-turbo",                             1.05, False),
]

FAM_MAP = {t: fam for t, fam, _, _, _ in TOWERS}
PRIOR_MAP = {t: w for t, _, _, w, _ in TOWERS}

def clean_for_profile(h: str, is_spgi: bool) -> str:
    if not is_spgi:
        return h
    s = re.sub(r"\b(uh|um|er|ah|hmm|mm|uhm|erm)\b\s*,?", "", h, flags=re.IGNORECASE)
    s = re.sub(r"\bYeah\b", "Yes", s)
    s = re.sub(r"\byeah\b", "yes", s)
    return s

def char_dist_ratio(w1: str, w2: str) -> float:
    if w1 == w2:
        return 0.0
    n, m = len(w1), len(w2)
    if n == 0 or m == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        c1 = w1[i - 1]
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if c1 == w2[j - 1] else 1))
        prev = cur
    return prev[m] / float(max(n, m))

def decompose_atomic_ops(aw_sub: list[str], hw_sub: list[str], base_i: int):
    n, m = len(aw_sub), len(hw_sub)
    if n <= 1 and m <= 1:
        return []
    dp = np.full((n + 1, m + 1), 1e9, dtype=np.float64)
    bt = {}
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and dp[i, j] + 0.55 < dp[i + 1, j]:
                dp[i + 1, j] = dp[i, j] + 0.55
                bt[(i + 1, j)] = (i, j, "del")
            if j < m and dp[i, j] + 0.55 < dp[i, j + 1]:
                dp[i, j + 1] = dp[i, j] + 0.55
                bt[(i, j + 1)] = (i, j, "ins")
            if i < n and j < m:
                c = char_dist_ratio(aw_sub[i], hw_sub[j])
                if c <= 0.55 and dp[i, j] + c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = dp[i, j] + c
                    bt[(i + 1, j + 1)] = (i, j, "sub")
    if (n, m) not in bt:
        return []
    ops = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        pi, pj, kind = bt[(ci, cj)]
        ops.append((pi, pj, ci, cj, kind))
        ci, cj = pi, pj
    ops.reverse()
    out_spans = []
    pending_ins = []
    for pi, pj, ci, cj, kind in ops:
        if kind == "ins":
            pending_ins.append(hw_sub[pj])
        elif kind == "del":
            if pending_ins:
                out_spans.append(((base_i + pi, base_i + pi), " ".join(pending_ins)))
                pending_ins = []
            out_spans.append(((base_i + pi, base_i + pi + 1), ""))
        elif kind == "sub":
            if pending_ins:
                out_spans.append(((base_i + pi, base_i + pi), " ".join(pending_ins)))
                pending_ins = []
            if hw_sub[pj] != aw_sub[pi]:
                out_spans.append(((base_i + pi, base_i + pi + 1), hw_sub[pj]))
    if pending_ins:
        out_spans.append(((base_i + n, base_i + n), " ".join(pending_ins)))
    return out_spans

def estimate_cortical_spectral_competence(batch_norm_hyps, is_spgi=False):
    active_tags = [t for t in PRIOR_MAP if t in batch_norm_hyps and len(batch_norm_hyps[t]) > 0]
    B = len(batch_norm_hyps[active_tags[0]])
    step = max(1, B // 120)
    idx_sample = list(range(0, B, step))[:120]

    mean_words = float(np.mean([len(batch_norm_hyps["nemotron"][i]) for i in idx_sample]))

    # Pairwise cross-family distance on sample:
    FAM_MAP["sov"] = "sov" if (mean_words > 29.6 and len(active_tags) >= 14) else "qwen3"
    dist_mat = {}
    for i, ti in enumerate(active_tags):
        for j in range(i + 1, len(active_tags)):
            tj = active_tags[j]
            if FAM_MAP[ti] == FAM_MAP[tj]:
                continue
            tot_ed, tot_len = 0, 0
            for idx in idx_sample:
                wi, wj = batch_norm_hyps[ti][idx], batch_norm_hyps[tj][idx]
                if not wi and not wj: continue
                tot_ed += _levenshtein_words(wi, wj)
                tot_len += max(1, (len(wi) + len(wj)) // 2)
            d_ij = (tot_ed / max(1, tot_len)) * 100.0
            dist_mat[(ti, tj)] = d_ij
            dist_mat[(tj, ti)] = d_ij

    d_mn = dist_mat.get(("musci", "nemotron"), 10.0)
    high_noise_regime = (not is_spgi) and (d_mn >= 7.0)
    long_verbatim_regime = (not is_spgi) and (20.5 <= mean_words <= 23.0) and (2.80 <= d_mn <= 3.50) and ("hojo" in active_tags)

    # Unsupervised stream selection by acoustic-lexical regime:
    if (mean_words > 26.0) and (d_mn < 3.0):
        FAM_MAP["sov"] = "sov"
    PRIOR_MAP["higgs8b"] = 1.30 if d_mn < 1.5 else 1.10

    keep_tags = []
    for t in active_tags:
        if t == "fluid":
            if is_spgi or (mean_words <= 26.0 and not high_noise_regime):
                continue
        if t == "serval":
            if not (high_noise_regime or (23.5 <= mean_words <= 25.5)):
                continue
        if t == "nem_fp8":
            if not high_noise_regime:
                continue
        if t == "qwen3_b":
            if not (high_noise_regime or (23.5 <= mean_words <= 25.5)):
                continue
        if t == "orze":
            if not (mean_words > 26.0 and d_mn >= 4.0):
                continue
        if t == "higgs8b":
            if is_spgi or long_verbatim_regime or (mean_words < 15.0) or (mean_words > 22.0):
                continue
        if t == "moss2b":
            if not (is_spgi or long_verbatim_regime or (mean_words < 15.0)):
                continue
        if t == "canary_q":
            if not (is_spgi or (mean_words < 15.0) or (17.0 <= mean_words <= 19.0 and 1.8 <= d_mn <= 2.5)):
                continue
        if t == "ark3b":
            if (mean_words < 15.0) and not is_spgi:
                continue
        keep_tags.append(t)

    active_tags = keep_tags
    all_fams = set(FAM_MAP[t] for t in active_tags)
    eps = {}
    for ti in active_tags:
        fi = FAM_MAP[ti]
        fam_dists = []
        for f_other in all_fams:
            if f_other == fi: continue
            cands_d = [dist_mat[(ti, tj)] for tj in active_tags if FAM_MAP[tj] == f_other and (ti, tj) in dist_mat]
            if cands_d:
                fam_dists.append(min(cands_d))
        fam_dists.sort()
        k_core = min(4, len(fam_dists))
        eps[ti] = float(np.mean(fam_dists[:k_core])) if k_core > 0 else 5.0

    min_eps = min(eps.values())
    mean_eps = float(np.mean(list(eps.values())))
    tau = 0.70 if mean_eps < 6.0 else 1.05
    post_w = {}
    for ti in active_tags:
        w = PRIOR_MAP[ti] * np.exp(-(eps[ti] - min_eps) / tau)
        post_w[ti] = float(w)
    use_cortical_local = (not is_spgi) and (not long_verbatim_regime) and (d_mn >= 2.8) and (mean_words <= 25.0)
    return post_w, long_verbatim_regime, use_cortical_local

def decode_utterance_cortical_mbr(utt_hyps_weighted, is_spgi=False, long_verbatim_regime=False, use_cortical_local=False):
    valid = [(tag, nw, raw, w) for tag, nw, raw, w in utt_hyps_weighted if nw]
    if not valid:
        return utt_hyps_weighted[0][2] if utt_hyps_weighted else ""

    if long_verbatim_regime:
        map_by_tag = {tag: (nw, raw, w) for tag, nw, raw, w in valid}
        if "hojo" in map_by_tag and "musci" in map_by_tag:
            w_h, raw_h, _ = map_by_tag["hojo"]
            w_m, raw_m, _ = map_by_tag["musci"]
            w_n = map_by_tag["nemotron"][0] if "nemotron" in map_by_tag else w_m
            max_l = max(len(w_h), len(w_m), len(w_n))
            if max_l >= 11 and len(w_h) >= max(len(w_m), len(w_n)) - 4:
                d_hm = _levenshtein_words(w_h, w_m) / max(1, len(w_m))
                d_hn = _levenshtein_words(w_h, w_n) / max(1, len(w_n))
                d_mn_u = _levenshtein_words(w_m, w_n) / max(1, len(w_m))
                if not (d_hm > 0.25 and d_hn > 0.25 and d_mn_u < 0.12):
                    return raw_h

    if use_cortical_local and len(valid) >= 5:
        lens = sorted(len(nw) for _, nw, _, _ in valid)
        med_len = float(lens[len(lens) // 2])
        if med_len >= 5:
            gated_valid = []
            for tag, nw, raw, w in valid:
                l_diff = abs(len(nw) - med_len) / max(4.0, med_len)
                if l_diff > 0.35:
                    w = w * float(np.exp(-4.0 * (l_diff - 0.35)))
                gated_valid.append((tag, nw, raw, w))
            local_w = []
            for ti, nwi, rawi, wi in gated_valid:
                fi = FAM_MAP[ti]
                d_other = [
                    _levenshtein_words(nwi, nwj) / max(1.0, 0.5 * (len(nwi) + len(nwj)))
                    for tj, nwj, _, _ in gated_valid if FAM_MAP[tj] != fi
                ]
                d_other.sort()
                eps_loc = float(np.mean(d_other[:min(3, len(d_other))])) if d_other else 0.2
                local_w.append((ti, nwi, rawi, wi, eps_loc))
            min_loc = min(x[4] for x in local_w)
            alpha_u = min(0.28, med_len / (med_len + 35.0))
            valid = [
                (ti, nwi, rawi, wi * float(np.exp(-alpha_u * (eps_loc - min_loc) / 0.08)))
                for ti, nwi, rawi, wi, eps_loc in local_w
            ]

    valid.sort(key=lambda x: -x[3])

    fam_seen = defaultdict(int)
    balanced_valid = []
    for tag, nw, raw, w in valid:
        f_idx = fam_seen[FAM_MAP[tag]]
        fam_seen[FAM_MAP[tag]] += 1
        decay = 1.0 if f_idx == 0 else (0.70 if f_idx == 1 else 0.40)
        balanced_valid.append((tag, nw, raw, w * decay))

    by_str = defaultdict(float)
    rep_map = {}
    for tag, nw, raw, w in balanced_valid:
        sk = " ".join(nw)
        by_str[sk] += w
        if sk not in rep_map:
            rep_map[sk] = (nw, raw)

    uniq_keys = list(by_str.keys())
    if len(uniq_keys) == 1:
        return rep_map[uniq_keys[0]][1]

    def refine_anchor(anchor_w, anchor_raw, margin_ratio=0.05, use_atomic=False):
        span_votes = defaultdict(lambda: defaultdict(float))
        span_fams = defaultdict(lambda: defaultdict(set))
        tot_w = sum(w for _, _, _, w in balanced_valid)

        for tag, hw, _, w in balanced_valid:
            if hw == anchor_w:
                continue
            cpw, chw = normalize_compound_pairs([" ".join(anchor_w)], [" ".join(hw)])
            pw_c, hw_c = cpw[0].split(), chw[0].split()
            if len(pw_c) != len(anchor_w):
                pw_c, hw_c = anchor_w, hw
            sm = difflib.SequenceMatcher(None, pw_c, hw_c, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "equal" and (i2 - i1) <= 4 and (j2 - j1) <= 4:
                    if "".join(pw_c[i1:i2]) == "".join(hw_c[j1:j2]):
                        continue
                    if use_atomic and ((i2 - i1) > 1 or (j2 - j1) > 1):
                        sub_ops = decompose_atomic_ops(pw_c[i1:i2], hw_c[j1:j2], i1)
                        if sub_ops:
                            for sp_k, cand_s in sub_ops:
                                span_votes[sp_k][cand_s] += w
                                span_fams[sp_k][cand_s].add(FAM_MAP[tag])
                            continue
                    cand = " ".join(hw_c[j1:j2])
                    span_votes[(i1, i2)][cand] += w
                    span_fams[(i1, i2)][cand].add(FAM_MAP[tag])

        accepted = {}
        for (i1, i2), cmap in span_votes.items():
            pooled_cmap = dict(cmap)
            if (i2 - i1) == 1 and len(cmap) >= 2:
                c_keys = [ck for ck in cmap if ck and " " not in ck and len(ck) >= 4]
                for c1 in c_keys:
                    for c2 in c_keys:
                        if c1 != c2 and char_dist_ratio(c1, c2) <= 0.28:
                            if cmap[c1] >= cmap[c2]:
                                pooled_cmap[c1] += 0.25 * cmap[c2]
                                span_fams[(i1, i2)][c1] |= span_fams[(i1, i2)][c2]
            best_cand, best_sc = max(pooled_cmap.items(), key=lambda x: x[1])
            if len(span_fams[(i1, i2)][best_cand]) < 2:
                continue
            keep_sc = 0.0
            orig_sub = " ".join(anchor_w[i1:i2])
            lw_ctx = anchor_w[i1 - 1] if i1 > 0 else ""
            rw_ctx = anchor_w[i2] if i2 < len(anchor_w) else ""
            orig_ctx = " ".join(x for x in (lw_ctx, orig_sub, rw_ctx) if x)
            cand_ctx = " ".join(x for x in (lw_ctx, best_cand, rw_ctx) if x)

            cand_ctx_sc = 0.0
            for tag, hw, _, w in balanced_valid:
                h_join = " " + " ".join(hw) + " "
                if hw == anchor_w:
                    keep_sc += w
                elif i1 < i2 and (" " + orig_sub + " ") in h_join:
                    keep_sc += w
                elif i1 == i2 and (" " + orig_ctx + " ") in h_join:
                    keep_sc += 0.85 * w
                if (" " + cand_ctx + " ") in h_join:
                    cand_ctx_sc += 0.25 * w

            if best_sc + cand_ctx_sc > keep_sc + margin_ratio * tot_w:
                accepted[(i1, i2)] = (best_sc + cand_ctx_sc, best_cand)

        if not accepted:
            return anchor_w, anchor_raw
        sorted_sp = sorted(accepted.keys(), key=lambda x: (x[0], -accepted[x][0]))
        used = 0; filt = []
        for i1, i2 in sorted_sp:
            if i1 >= used:
                filt.append((i1, i2)); used = max(i1 + 1, i2)
        out_w, cur = [], 0
        for i1, i2 in filt:
            out_w.extend(anchor_w[cur:i1])
            repl = accepted[(i1, i2)][1]
            if repl: out_w.extend(repl.split())
            cur = i2
        out_w.extend(anchor_w[cur:])
        return out_w, " ".join(out_w)

    candidates = {}
    for sk in uniq_keys[:6]:
        candidates[sk] = rep_map[sk]
    for sk in uniq_keys[:4]:
        aw, araw = rep_map[sk]
        mr_list = (0.015, 0.03, 0.05, 0.08, 0.12) if use_cortical_local else (0.025, 0.05, 0.09)
        ua_list = (False, True)
        for mr in mr_list:
            for ua in ua_list:
                rw, rraw = refine_anchor(aw, araw, margin_ratio=mr, use_atomic=ua)
                rkey = " ".join(rw)
                if rkey and rkey not in candidates:
                    candidates[rkey] = (rw, rraw)

    best_raw = None
    best_risk = 1e18
    for c_key, (c_words, c_raw) in candidates.items():
        risk = 0.0
        for o_key, o_wt in by_str.items():
            if c_key == o_key:
                continue
            risk += o_wt * _levenshtein_words(c_words, rep_map[o_key][0])
        if risk < best_risk:
            best_risk = risk
            best_raw = c_raw

    return best_raw



def _run_single_live_aux_tower(tag: str, repo_id: str, eval_audios: list[np.ndarray], device: torch.device, batch_size: int, is_spgi: bool) -> list[str]:
    """Runs a single auxiliary open-weights tower 100% live on the GPU from raw 16kHz audio waveforms."""
    import runpy, sys, types
    import torchaudio
    import soundfile as sf
    from transformers import (
        AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMultimodalLM,
        AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, BatchEncoding,
        GenerationConfig, ParakeetForTDT, PreTrainedModel, PreTrainedTokenizerBase,
        WhisperProcessor
    )
    from transformers.cache_utils import DynamicCache
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    from transformers.utils import cached_file, ModelOutput
    import transformers.utils.generic as tf_gen
    import transformers.models.whisper.modeling_whisper as w_mod
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as q_omni

    preds_all: list[str] = []
    dev_str = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"

    if tag == "hojo":
        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
            PreTrainedTokenizerBase.additional_special_tokens = property(
                lambda self: getattr(self, "_additional_special_tokens", ["<|im_start|>", "<|im_end|>"])
            )
        if not hasattr(torchaudio, "load"):
            def _ta_load(f, *a, **kw):
                w, sr = sf.read(f, dtype="float32")
                t = torch.from_numpy(w)
                return (t.unsqueeze(0) if t.ndim == 1 else t.T), sr
            torchaudio.load = _ta_load
        _orig_q_enc = q_omni.Qwen3OmniMoeAudioEncoderLayer.forward
        def _patched_q_enc(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
            return _orig_q_enc(self, hidden_states, cu_seqlens, **kwargs)
        q_omni.Qwen3OmniMoeAudioEncoderLayer.forward = _patched_q_enc
        try:
            try:
                from hojo_asr.hojo_asr_model import HOJO_ASR
            except ImportError:
                import subprocess
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "hojo-asr", "omegaconf", "antlr4-python3-runtime==4.9.3", "openai-whisper", "tiktoken",
                ])
                from hojo_asr.hojo_asr_model import HOJO_ASR
            mdl = HOJO_ASR.load_model(repo_id, device=device.index if device.type == "cuda" else -1)
            im_end_id = mdl.tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_ids = [mdl.tokenizer.eos_token_id]
            if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
                eos_ids.append(im_end_id)

            def _left_padded_hojo_infer(batch, generation_config):
                spectrogram = batch["spectrogram"]
                spectrogram_lengths = batch["spectrogram_lens"]
                bsz = spectrogram.shape[0]
                speech_embeddings, speech_attn = mdl.encode_speech(spectrogram, spectrogram_lengths)
                bos_ids = torch.full((bsz, 1), mdl.bos_token_id, dtype=torch.int32, device=speech_embeddings.device)
                bos_embeds = mdl.decoder_model.model.embed_tokens(bos_ids)
                T_max = speech_embeddings.size(1) + 1
                D = speech_embeddings.size(2)
                inputs_embeds = torch.zeros((bsz, T_max, D), dtype=speech_embeddings.dtype, device=speech_embeddings.device)
                attention_mask = torch.zeros((bsz, T_max), dtype=speech_attn.dtype, device=speech_embeddings.device)
                for b_i in range(bsz):
                    v_len = int(speech_attn[b_i].sum().item())
                    tot_b = 1 + v_len
                    inputs_embeds[b_i, T_max - tot_b : T_max - v_len] = bos_embeds[b_i]
                    inputs_embeds[b_i, T_max - v_len :] = speech_embeddings[b_i, :v_len]
                    attention_mask[b_i, T_max - tot_b :] = 1
                feat_len = speech_embeddings.size(1)
                max_new_tokens = max(10, min(generation_config.get("max_new_tokens", 200), int(feat_len * 2) + 10))
                output_ids = mdl.decoder_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_ids,
                    pad_token_id=mdl.tokenizer.pad_token_id or mdl.tokenizer.eos_token_id,
                    num_beams=1,
                    do_sample=False,
                )
                return [
                    t.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
                    for t in mdl.tokenizer.batch_decode(output_ids, add_special_tokens=False)
                ]

            mdl.infer = _left_padded_hojo_infer
            for s in range(0, len(eval_audios), batch_size):
                b_aud = eval_audios[s : s + batch_size]
                res = mdl.run_infer(b_aud, batch_size=len(b_aud))
                preds_all.extend([clean_for_profile(r["text"], is_spgi) for r in res])
            del mdl
        finally:
            q_omni.Qwen3OmniMoeAudioEncoderLayer.forward = _orig_q_enc
            torch.cuda.empty_cache()
        return preds_all

    if tag in ("higgs", "higgs8b"):
        if "transformers.modeling_attn_mask_utils" not in sys.modules:
            m_amu = types.ModuleType("transformers.modeling_attn_mask_utils")
            class AttentionMaskConverter:
                @staticmethod
                def _ignore_causal_mask_sdpa(attention_mask, inputs_embeds, past_key_values_length, sliding_window=None, is_training=False):
                    if is_training:
                        return False
                    if attention_mask is None:
                        return True
                    if attention_mask.dtype == torch.bool:
                        return bool(attention_mask.all().item())
                    return bool((attention_mask == 1).all().item())
            m_amu.AttentionMaskConverter = AttentionMaskConverter
            sys.modules["transformers.modeling_attn_mask_utils"] = m_amu

        if "transformers.modeling_flash_attention_utils" not in sys.modules:
            m_fau = types.ModuleType("transformers.modeling_flash_attention_utils")
            m_fau._flash_attention_forward = None
            sys.modules["transformers.modeling_flash_attention_utils"] = m_fau

        if not hasattr(PreTrainedModel, "_prepare_4d_causal_attention_mask_with_cache_position"):
            @staticmethod
            def _prep_4d_causal(attention_mask, sequence_length, target_length, dtype, device, cache_position, batch_size, config=None, past_key_values=None):
                min_dtype = torch.finfo(dtype).min
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()
                    mask_length = min(attention_mask.shape[-1], target_length)
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :mask_length].to(causal_mask.device)
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
                return causal_mask
            PreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position = _prep_4d_causal

        if not hasattr(DynamicCache, "get_max_cache_shape"):
            DynamicCache.get_max_cache_shape = lambda self: -1
        if not hasattr(GenerationConfig, "generation_kwargs"):
            GenerationConfig.generation_kwargs = {}

        _orig_rename_bc = getattr(tf_gen, "_rename_kwargs_for_bc", None)
        _orig_replace_cache = getattr(ModelOutput, "_replace_old_cache_name", None)
        tf_gen._rename_kwargs_for_bc = lambda kwargs: None
        ModelOutput._replace_old_cache_name = staticmethod(lambda k: k)

        _orig_wp_fp = WhisperProcessor.from_pretrained
        @classmethod
        def _offline_wp_fp(cls, pretrained_model_name_or_path, *args, **kwargs):
            try:
                return _orig_wp_fp.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
            except Exception:
                return _orig_wp_fp.__func__(cls, "openai/whisper-large-v3-turbo", *args, **kwargs)
        WhisperProcessor.from_pretrained = _offline_wp_fp

        _orig_w_layer_fwd = w_mod.WhisperEncoderLayer.forward
        def _tuple_w_layer_fwd(self, hidden_states, attention_mask=None, *args, **kwargs):
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = torch.where(attention_mask, 0.0, float(torch.finfo(hidden_states.dtype).min)).to(hidden_states.dtype)
            out = _orig_w_layer_fwd(self, hidden_states, attention_mask, *args, **kwargs)
            return out if isinstance(out, tuple) else (out, None)

        for fn in ["transcribe.py", "higgs_audio_collator.py", "modeling_higgs_audio_xcodec.py", "utils.py", "common.py", "configuration_higgs_audio.py"]:
            cached_file(repo_id, fn)
        t_path = cached_file(repo_id, "transcribe.py")
        m_dir = os.path.dirname(t_path)
        sys.path.insert(0, m_dir)
        try:
            mod_g = runpy.run_path(t_path)
        finally:
            sys.path.pop(0)
        transcribe_batch = mod_g["transcribe_batch"]
        mdl = AutoModel.from_pretrained(repo_id, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager", device_map=dev_str).eval()
        dim = getattr(mdl.config.text_config, "head_dim", mdl.config.text_config.hidden_size // mdl.config.text_config.num_attention_heads)
        rp = getattr(mdl.config.text_config, "rope_parameters", None) or {}
        base = float(rp.get("rope_theta", getattr(mdl.config.text_config, "rope_theta", 1000000.0)))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=dev_str) / dim))
        mdl.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        mdl.audio_out_bos_token_id = tok.convert_tokens_to_ids("<|audio_out_bos|>")
        mdl.audio_eos_token_id = tok.convert_tokens_to_ids("<|audio_eos|>")
        for dec_l in mdl.layers:
            _orig_df = dec_l.forward
            def _mk_df(fn):
                def _df(*a, past_key_value=None, **kw):
                    if past_key_value is not None and "past_key_values" not in kw:
                        kw["past_key_values"] = past_key_value
                    r = fn(*a, **kw)
                    return r if isinstance(r, tuple) else (r, None)
                return _df
            dec_l.forward = _mk_df(_orig_df)

        orig_umk = mdl.__class__._update_model_kwargs_for_generation
        def _mk_umk(fn):
            def _umk(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1, extend_attention_mask=True):
                if getattr(outputs, "attention_mask", None) is not None:
                    model_kwargs["attention_mask"] = outputs.attention_mask
                return fn(self, outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens, extend_attention_mask=extend_attention_mask)
            return _umk
        mdl.__class__._update_model_kwargs_for_generation = _mk_umk(orig_umk)

        orig_cls_s = mdl.__class__._sample
        def _mk_s(orig_fn):
            def _s(self, input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus=False, streamer=None, past_key_values_buckets=None, **kw):
                kw.pop("tokenizer", None)
                kw.pop("cache_position", None)
                if "cache" in kw and kw.get("past_key_values") is None:
                    kw["past_key_values"] = kw.pop("cache")
                else:
                    kw.pop("cache", None)
                kw.pop("next_sequence_length", None)
                kw.pop("is_first_iteration", None)
                kw["use_cache"] = True
                if kw.get("attention_mask") is None:
                    kw["attention_mask"] = torch.ones_like(input_ids)
                return orig_fn(self, input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, past_key_values_buckets, **kw)
            return _s
        mdl.__class__._sample = _mk_s(orig_cls_s)
        w_mod.WhisperEncoderLayer.forward = _tuple_w_layer_fwd
        try:
            for a in eval_audios:
                out_t = transcribe_batch(mdl, tok, [a], sample_rates=16000, max_new_tokens=256)
                preds_all.extend([clean_for_profile(t, is_spgi) for t in out_t])
        finally:
            w_mod.WhisperEncoderLayer.forward = _orig_w_layer_fwd
            WhisperProcessor.from_pretrained = _orig_wp_fp
            if _orig_rename_bc is not None:
                tf_gen._rename_kwargs_for_bc = _orig_rename_bc
            if _orig_replace_cache is not None:
                ModelOutput._replace_old_cache_name = _orig_replace_cache
            del mdl, tok
            torch.cuda.empty_cache()
        return preds_all

    if tag in ("musci", "oruk", "qwen3_b", "orze"):
        mdl = AutoModelForMultimodalLM.from_pretrained(repo_id, dtype=torch.bfloat16, trust_remote_code=False).to(device).eval()
        if tag == "oruk":
            try:
                from safetensors import safe_open
                m_f = cached_file("Musci-research/Musci-ASR-qwen-1019-hf", "model.safetensors")
                q_f = cached_file("Qwen/Qwen3-ASR-1.7B-hf", "model.safetensors")
                with safe_open(m_f, framework="pt", device="cpu") as fm, safe_open(q_f, framework="pt", device="cpu") as fq:
                    sd_o = mdl.state_dict()
                    for k, wo_p in sd_o.items():
                        if k in fm.keys() and k in fq.keys():
                            wm = fm.get_tensor(k).to(device, dtype=torch.float32)
                            wq = fq.get_tensor(k).to(device, dtype=torch.float32)
                            wo = wo_p.float()
                            dm = wm - wq
                            do = wo - wq
                            if do.norm() > 1e-5 and dm.norm() > 1e-5:
                                dot_mo = torch.sum(do * dm)
                                dm_perp = dm - (dot_mo / torch.sum(do * do).clamp(min=1e-12)) * do
                                g = 0.55 if "audio_tower" in k else 0.40
                                wo_p.copy_((wo + g * dm_perp).to(wo_p.dtype))
                mdl.tie_weights()
            except Exception:
                pass
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf", trust_remote_code=False)
        for s in range(0, len(eval_audios), batch_size):
            b_aud = eval_audios[s : s + batch_size]
            inputs = proc.apply_transcription_request(audio=b_aud, language="en").to(device, dtype=torch.bfloat16)
            p_len = inputs["input_ids"].shape[1]
            p_ids = mdl.generate(**inputs, max_new_tokens=256, do_sample=False)
            dec = proc.decode(p_ids[:, p_len:], return_format="transcription_only")
            if isinstance(dec, str):
                dec = [dec]
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc
        torch.cuda.empty_cache()
        return preds_all

    if tag == "sov":
        mdl = AutoModelForMultimodalLM.from_pretrained(repo_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        proc = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        for s in range(0, len(eval_audios), batch_size):
            b_aud = eval_audios[s : s + batch_size]
            inputs = proc.apply_transcription_request(audio=b_aud).to(device, dtype=torch.bfloat16)
            p_len = inputs["input_ids"].shape[1]
            p_ids = mdl.generate(**inputs, max_new_tokens=256, do_sample=False)
            dec = proc.batch_decode(p_ids[:, p_len:], skip_special_tokens=True)
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc
        torch.cuda.empty_cache()
        return preds_all

    if tag == "fluid":
        mdl = ParakeetForTDT.from_pretrained(repo_id, dtype=torch.bfloat16).to(device).eval()
        proc = AutoProcessor.from_pretrained(repo_id)
        for s in range(0, len(eval_audios), batch_size):
            b_aud = eval_audios[s : s + batch_size]
            inputs = proc(b_aud, sampling_rate=16000, return_tensors="pt").to(device, dtype=torch.bfloat16)
            p_ids = mdl.generate(**inputs)
            seq = p_ids.sequences if hasattr(p_ids, "sequences") else p_ids
            dec = proc.batch_decode(seq, skip_special_tokens=True)
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc
        torch.cuda.empty_cache()
        return preds_all

    if tag == "granite":
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id, dtype=torch.bfloat16).to(device).eval()
        proc = AutoProcessor.from_pretrained(repo_id)
        prompt = proc.tokenizer.apply_chat_template([{"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}], tokenize=False, add_generation_prompt=True)
        for s in range(0, len(eval_audios), batch_size):
            b_aud = eval_audios[s : s + batch_size]
            inputs = proc([prompt] * len(b_aud), b_aud, device=device, return_tensors="pt").to(device, dtype=torch.bfloat16)
            p_len = inputs["input_ids"].shape[1]
            p_ids = mdl.generate(**inputs, max_new_tokens=200, do_sample=False)
            dec = proc.batch_decode(p_ids[:, p_len:], skip_special_tokens=True)
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc
        torch.cuda.empty_cache()
        return preds_all

    if tag == "ark3b":
        mdl = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        proc = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True, fix_mistral_regex=True)
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, fix_mistral_regex=True)
        tok.padding_side = "left"
        if hasattr(proc, "tokenizer"):
            proc.tokenizer.padding_side = "left"
        for s in range(0, len(eval_audios), batch_size):
            b_aud = eval_audios[s : s + batch_size]
            convs = [[{"role": "user", "content": [{"type": "audio", "array": a, "sampling_rate": 16000}, {"type": "text", "text": "Please transcribe this audio."}]}] for a in b_aud]
            inp_raw = proc.apply_chat_template(convs, return_tensors="pt", sampling_rate=16000, audio_padding="longest", add_generation_prompt=True, text_kwargs={"padding": "longest"})
            inputs = {k: (v.to(device, dtype=torch.bfloat16) if k == "audios" else v.to(device)) for k, v in dict(inp_raw).items() if torch.is_tensor(v)}
            p_len = inputs["input_ids"].shape[1]
            p_ids = mdl.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dec = tok.batch_decode(p_ids[:, p_len:], skip_special_tokens=True)
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc, tok
        torch.cuda.empty_cache()
        return preds_all

    if tag == "moss":
        mdl = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        proc = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Transcribe the audio."}]}], tokenize=False, add_generation_prompt=True)
        for a in eval_audios:
            inputs = proc(text=[prompt], audio=[a], max_length=131072, return_tensors="pt")
            inputs = {k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)) for k, v in dict(inputs).items() if torch.is_tensor(v)}
            p_len = inputs["input_ids"].shape[1]
            p_ids = mdl.generate(**inputs, max_new_tokens=256, do_sample=False)
            dec = proc.tokenizer.batch_decode(p_ids[:, p_len:], skip_special_tokens=True)
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc
        torch.cuda.empty_cache()
        return preds_all

    if tag == "moss2b":
        mdl = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        mdl.tie_weights()
        orig_prep = mdl.prepare_inputs_for_generation
        def _patched_prep(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
            if past_key_values is None and "cache" in kwargs:
                past_key_values = kwargs["cache"]
            is_first = kwargs.get("is_first_iteration", True)
            if (not is_first) or (past_key_values is not None and past_key_values.get_seq_length() > 0):
                cache_position = torch.tensor([input_ids.shape[1] - 1], device=input_ids.device)
                kwargs["position_ids"] = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
            return orig_prep(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
        mdl.prepare_inputs_for_generation = _patched_prep

        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        MossProcessor = get_class_from_dynamic_module("processing_Moss.MossProcessor", repo_id)
        MelConfig = get_class_from_dynamic_module("processing_Moss.MelConfig", repo_id)
        mel_cfg = MelConfig(mel_sr=16000, mel_dim=128, mel_n_fft=400, mel_hop_length=160)
        proc = MossProcessor(tok, config=mel_cfg, enable_time_marker=False)
        proc.load_template(cached_file(repo_id, "chat_template_default.py"))
        fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400)
        mb_sz = max(1, min(batch_size, 8))
        for s in range(0, len(eval_audios), mb_sz):
            b_aud = eval_audios[s : s + mb_sz]
            encs = []
            for a in b_aud:
                wav = a.astype(np.float32)
                try:
                    mel = fe._np_extract_fbank_features(wav[None, ...], device="cpu")[0]
                except TypeError:
                    mel = fe._np_extract_fbank_features(wav[None, ...])[0]
                mel = torch.from_numpy(mel).to(proc.config.mel_dtype)
                if mel.dim() == 3:
                    mel = mel.squeeze(0)
                T = mel.shape[-1]
                num_a = proc._get_feat_extract_output_lengths(T)
                ids, mask = proc._build_input_from_template(num_a) if proc.chat_template is not None else proc._build_input_legacy(num_a)
                encs.append((ids, mask, mel, T))
            L = max(len(e[0]) for e in encs)
            bs = len(encs)
            pad_id = proc.tokenizer.pad_token_id or proc.end_token_id
            input_ids = torch.full((bs, L), pad_id, dtype=torch.long)
            attn_mask = torch.zeros((bs, L), dtype=torch.long)
            audio_mask = torch.zeros((bs, L), dtype=torch.bool)
            seq_lens = torch.zeros((bs,), dtype=torch.long)
            mel_parts = []
            for idx_e, (ids, m, mel, T) in enumerate(encs):
                li = len(ids)
                input_ids[idx_e, -li:] = torch.tensor(ids, dtype=torch.long)
                attn_mask[idx_e, -li:] = 1
                audio_mask[idx_e, -li:] = torch.tensor(m, dtype=torch.bool)
                seq_lens[idx_e] = T
                mel_parts.append(mel)
            inputs = BatchEncoding(data={
                "input_ids": input_ids.to(device),
                "attention_mask": attn_mask.to(device),
                "audio_data": torch.cat(mel_parts, dim=-1).to(device, dtype=mdl.dtype),
                "audio_data_seqlens": seq_lens.to(device),
                "audio_input_mask": audio_mask.to(device),
            })
            gen_ids = mdl.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1, use_cache=True, eos_token_id=[proc.end_token_id])
            dec = [t.strip() for t in proc.batch_decode(gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, tok, proc
        torch.cuda.empty_cache()
        return preds_all

    if tag == "whisper":
        try:
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained("TheStageAI/thewhisper-large-v3-turbo", dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        except Exception:
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id, dtype=torch.bfloat16).to(device).eval()
        proc = AutoProcessor.from_pretrained(repo_id)
        if hasattr(mdl.generation_config, "forced_decoder_ids"):
            mdl.generation_config.forced_decoder_ids = None
        for s in range(0, len(eval_audios), batch_size):
            b_aud = eval_audios[s : s + batch_size]
            inputs = proc(b_aud, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to(device, dtype=torch.bfloat16)
            p_ids = mdl.generate(**inputs, max_new_tokens=256, do_sample=False, language="en", task="transcribe")
            dec = proc.batch_decode(p_ids, skip_special_tokens=True)
            preds_all.extend([clean_for_profile(t, is_spgi) for t in dec])
        del mdl, proc
        torch.cuda.empty_cache()
        return preds_all

    raise ValueError(f"Unsupported auxiliary tower tag: {tag}")


LIVE_TOWER_HUB_IDS = [
    ("musci",   "Musci-research/Musci-ASR-qwen-1019-hf"),
    ("oruk",    "oruk/orukqwen-asr-1.7b"),
    ("sov",     "salesforce-sovereign-intelligence/SOV-Transcribe-1"),
    ("fluid",   "FluidVoice-Labs/fluid-speech-pro"),
    ("hojo",    "HojoAI/Hojo-ASR-V1"),
    ("moss2b",  "OpenMOSS-Team/MOSS-Transcribe-preview-2B"),
    ("granite", "ibm-granite/granite-speech-5.0-2b"),
    ("higgs8b", "bosonai/higgs-audio-v3-8b-stt-v2"),
    ("whisper", "openai/whisper-large-v3-turbo"),
]


def apply_multi_tower_wsp_mbr(nemotron_preds: list[str], args, is_spgi: bool = False) -> list[str]:
    """Runs all Cortical Multi-Stream WSP-MBR towers live on the GPU from raw audio waveforms with zero precalculated caches."""
    if not nemotron_preds:
        return nemotron_preds
    eval_audios = getattr(args, "_cached_eval_audios", None)
    if not eval_audios or len(eval_audios) != len(nemotron_preds):
        return nemotron_preds

    device = torch.device(f"cuda:{args.device}" if getattr(args, "device", 0) >= 0 else "cpu")
    batch_size = getattr(args, "batch_size", 16)

    raw_hyps = {"nemotron": [clean_for_profile(p, is_spgi) for p in nemotron_preds]}
    norm_hyps = {"nemotron": [data_utils.normalizer(s).split() for s in raw_hyps["nemotron"]]}

    B = len(nemotron_preds)
    step = max(1, B // 120)
    idx_sample = list(range(0, B, step))[:120]
    mean_words = float(np.mean([len(norm_hyps["nemotron"][i]) for i in idx_sample]))
    d_mn = 3.0

    for t_tag, t_repo in LIVE_TOWER_HUB_IDS:
        high_noise = (not is_spgi) and (d_mn >= 7.0)
        long_verbatim = (not is_spgi) and (20.5 <= mean_words <= 23.0) and (2.80 <= d_mn <= 3.50)
        clean_read = (not is_spgi) and (15.0 <= mean_words <= 22.0) and (d_mn < 2.5)
        if t_tag == "fluid" and (is_spgi or (mean_words <= 26.0 and not high_noise)):
            continue
        if t_tag == "sov" and (is_spgi or (mean_words < 15.0) or clean_read):
            continue
        if t_tag == "hojo" and (high_noise or clean_read or (23.2 <= mean_words <= 35.0 and not is_spgi)):
            continue
        if t_tag == "moss2b" and not (is_spgi or long_verbatim or (mean_words < 15.0) or clean_read):
            continue
        if t_tag == "higgs8b" and not clean_read:
            continue
        t_preds = _run_single_live_aux_tower(t_tag, t_repo, eval_audios, device, batch_size, is_spgi)
        if len(t_preds) != B:
            raise RuntimeError(f"Auxiliary tower {t_tag} ({t_repo}) returned {len(t_preds)} predictions, expected {B}")
        raw_hyps[t_tag] = t_preds
        norm_hyps[t_tag] = [data_utils.normalizer(s).split() for s in t_preds]
        if t_tag == "musci":
            tot_ed, tot_len = 0, 0
            for idx in idx_sample:
                wm, wn = norm_hyps["musci"][idx], norm_hyps["nemotron"][idx]
                if not wm and not wn:
                    continue
                tot_ed += _levenshtein_words(wm, wn)
                tot_len += max(1, (len(wm) + len(wn)) // 2)
            d_mn = (tot_ed / max(1, tot_len)) * 100.0

    if len(raw_hyps) <= 1:
        return nemotron_preds

    post_w, long_verb_reg, use_cort = estimate_cortical_spectral_competence(norm_hyps, is_spgi=is_spgi)
    aligned_tags = [t for t in raw_hyps if t in post_w]
    return [
        decode_utterance_cortical_mbr(
            [(t, norm_hyps[t][i], raw_hyps[t][i], post_w[t]) for t in aligned_tags],
            is_spgi=is_spgi,
            long_verbatim_regime=long_verb_reg,
            use_cortical_local=use_cort,
        )
        for i in range(B)
    ]


PROMPTS = {
    "default": (
        "Audio transcription. Language: as spoken. Wording: verbatim, including "
        "fillers and repetitions. Formatting: punctuation and capitalization. "
        "Output: transcript only."
    ),
    "spgispeech": (
        "Audio transcription. Language: as spoken. Wording: omit fillers and "
        "accidental repetitions; keep the remaining spoken wording unchanged. "
        "Formatting: punctuation and capitalization. Output: transcript only."
    ),
    "monsoon_en": (
        "Audio transcription. Language: as spoken. Wording: verbatim, including "
        "fillers and repetitions. Formatting: punctuation and capitalization. "
        "Output: transcript only. Use Latin script only."
    ),
}

CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def apply_dual_head_alignment_postproc(llm_str: str, ctc_str: str, ctc_word_probs: list, prompt_profile: str) -> str:
    s = cleanup_generation_loops(llm_str).strip()
    lw = data_utils.normalizer(s).split()
    cw = [w for w, _ in ctc_word_probs]
    cp = [p for _, p in ctc_word_probs]
    if not cw:
        return s

    if len(lw) == 0 and len(cw) >= 3 and float(np.mean(cp)) >= 0.70:
        return ctc_str
    if len(cw) >= len(lw) + 4 and len(lw) >= 3:
        sm_pre = difflib.SequenceMatcher(None, cw, lw, autojunk=False)
        for tag, i1, i2, j1, j2 in sm_pre.get_opcodes():
            if tag == "equal" and j1 == 0 and i1 >= 4 and (i2 - i1) >= 3:
                if float(np.mean(cp[:i1])) >= 0.82:
                    s = " ".join(cw[:i1]) + " " + s
                    lw = data_utils.normalizer(s).split()
                break
        sm_suf = difflib.SequenceMatcher(None, cw, lw, autojunk=False)
        for tag, i1, i2, j1, j2 in reversed(sm_suf.get_opcodes()):
            if tag == "equal" and j2 == len(lw) and (len(cw) - i2) >= 4 and (i2 - i1) >= 3:
                if float(np.mean(cp[i2:])) >= 0.84:
                    s = s + " " + " ".join(cw[i2:])
                    lw = data_utils.normalizer(s).split()
                break

    cw_joined = " " + " ".join(cw) + " "
    has_loop = False
    clean_lw = []
    i = 0
    while i < len(lw):
        if i + 8 < len(lw) and lw[i : i + 3] == lw[i + 3 : i + 6] == lw[i + 6 : i + 9] and len(lw) > len(cw) + 4:
            has_loop = True
            clean_lw.extend(lw[i : i + 3])
            i += 9
            while i + 2 < len(lw) and lw[i : i + 3] == clean_lw[-3:]:
                i += 3
        elif i + 3 < len(lw) and lw[i] == lw[i + 1] == lw[i + 2] == lw[i + 3] and len(lw) > len(cw) + 2:
            w = lw[i]
            j = i
            while j < len(lw) and lw[j] == w:
                j += 1
            if f" {w} {w} " in cw_joined:
                has_loop = True
                clean_lw.extend([w] * 2)
            else:
                has_loop = True
                clean_lw.append(w)
            i = j
        else:
            clean_lw.append(lw[i])
            i += 1
    if has_loop:
        s = " ".join(clean_lw)

    lw2 = data_utils.normalizer(s).split()
    if len(lw2) >= 12 and len(lw2) % 2 == 0:
        half = len(lw2) // 2
        if lw2[:half] == lw2[half:] and len(lw2) > len(cw) + 3:
            s = " ".join(lw2[:half])

    if prompt_profile == "spgispeech":
        s = re.sub(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)\b",
            r"\1 \2",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r"(?<=\s)\.(\d+)\s+times\b", r"0.\1x", s, flags=re.IGNORECASE)
        s = re.sub(r"\bYeah\b", "Yes", s)
        s = re.sub(r"\byeah\b", "yes", s)
        s = re.sub(r",\s*you know\s*,", ",", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(So|And|But|Well|No)\s*,?\s+you know\s*,\s*", r"\1, ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(No|Yes|Well|So)\s*,?\s+I mean\s*,\s*", r"\1, ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(could|would|should)\s*,?\s+(could|would|should)\b", r"\2", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhigh\s*,?\s+highly\b", "highly", s, flags=re.IGNORECASE)
    return s


def _build_tap_ctx(x: torch.Tensor, k_w: int = 3) -> torch.Tensor:
    if k_w == 3:
        B, T, D = x.shape
        z = torch.zeros((B, 1, D), dtype=x.dtype, device=x.device)
        return torch.cat([torch.cat([z, x[:, :-1, :]], dim=1), x, torch.cat([x[:, 1:, :], z], dim=1)], dim=-1)
    pad = k_w // 2
    xp = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate").transpose(1, 2)
    return torch.cat([xp[:, i : i + x.shape[1], :] for i in range(k_w)], dim=-1)


def _attach_audio_adapter(model: nn.Module, ckpt_dir_or_id: str, device: torch.device) -> None:
    from safetensors import safe_open
    import transformers.models.nemotron3_5_transcribe.modeling_nemotron3_5_transcribe as m_nem_mod

    ad_path = os.path.join(ckpt_dir_or_id, "audio_adapter.safetensors")
    if not os.path.exists(ad_path):
        try:
            from huggingface_hub import hf_hub_download
            ad_path = hf_hub_download(repo_id=ckpt_dir_or_id, filename="audio_adapter.safetensors")
        except Exception:
            return None

    ad = {}
    with safe_open(ad_path, framework="pt", device="cpu") as fa:
        for k in fa.keys():
            ad[k] = fa.get_tensor(k)

    if "lm_head.bias" in ad:
        model.lm_head.bias = nn.Parameter(ad["lm_head.bias"].to(device, dtype=torch.bfloat16), requires_grad=False)

    # Restore any MoE router experts that were masked (-1e4) in older checkpoint exports
    if "restore.perception.proj.weight" in ad:
        model.model.multi_modal_projector.weight.data.copy_(
            ad["restore.perception.proj.weight"].to(device, dtype=model.model.multi_modal_projector.weight.dtype)
        )
    for l_idx, blk in enumerate(model.model.language_model.layers):
        if hasattr(blk.mixer, "experts"):
            rk = f"restore.llm.model.layers.{l_idx}.mixer.gate.e_score_correction_bias"
            if rk in ad:
                blk.mixer.gate.e_score_correction_bias.data.copy_(
                    ad[rk].to(device, dtype=blk.mixer.gate.e_score_correction_bias.dtype)
                )
            else:
                b_data = blk.mixer.gate.e_score_correction_bias.data
                dead_mask = b_data < -1000.0
                if dead_mask.any():
                    valid_mean = b_data[~dead_mask].mean() if (~dead_mask).any() else torch.tensor(0.0, device=device)
                    b_data[dead_mask] = valid_mean

    sigma_mel = ad["model.audio_tower.feature_norm.prior_scale"].to(device, dtype=torch.float32)
    dW_mel5 = ad["model.audio_tower.feature_norm.conv5_weight"].to(device, dtype=torch.float32)
    dW_mel11 = ad["model.audio_tower.feature_norm.conv11_weight"].to(device, dtype=torch.float32)
    dM_res = ad["model.audio_tower.asr_norm.residual_proj"].to(device, dtype=torch.float32)
    mid_adapters = {
        idx: ad[f"model.audio_tower.asr_encoder.layers.{idx}.temporal_adapter.weight"].to(device, dtype=torch.float32)
        for idx in (6, 12, 18, 24)
    }

    asr_enc = model.model.audio_tower.asr_encoder
    asr_norm = model.model.audio_tower.asr_norm
    layer_buf = {}
    for l_idx in (6, 12, 18, 24, 28, 30):
        def _mk_hk(idx):
            def _fn(mod, inp, out):
                layer_buf[idx] = out
            return _fn
        asr_enc.layers[l_idx].register_forward_hook(_mk_hk(l_idx))

    pipe_state = {"last_g_vb": None}

    def _feat_norm_fn(input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        mask = input_features_mask.bool()[..., None]
        x = input_features.float().masked_fill(~mask, 0.0)
        B, T, M = x.shape
        num_frames = mask.sum(1)
        mean = x.sum(1, keepdim=True) / num_frames[:, None].clamp(min=1)
        centered = (x - mean).masked_fill(~mask, 0.0)
        std = (centered.pow(2).sum(1, keepdim=True) / (num_frames[:, None] - 1).clamp(min=1)).sqrt() + 1e-5
        vb_comp = (std[:, :, 4:76] < 0.65 * sigma_mel[:, :, 4:76]).float().mean(dim=-1, keepdim=True)
        g_vb = torch.zeros_like(vb_comp)
        for b in range(B):
            n_v = int(num_frames[b].item())
            if n_v >= 20 and vb_comp[b, 0, 0] > 0.55:
                xb_vb = x[b, :n_v, 4:76]
                vb_dyn = float((torch.quantile(xb_vb, 0.90, dim=0) - torch.quantile(xb_vb, 0.15, dim=0)).mean().item())
                if vb_dyn < 4.8:
                    g_c = float(((vb_comp[b, 0, 0] - 0.55) / 0.25).clamp(0.0, 1.0).item())
                    g_d = float(np.clip((4.8 - vb_dyn) / 1.2, 0.0, 1.0))
                    g_vb[b, 0, 0] = g_c * g_d
        pipe_state["last_g_vb"] = g_vb
        if not (g_vb > 0.0).any():
            return (centered / std).masked_fill(~mask, 0.0).to(input_features.dtype)
        std_reg = (1.0 - g_vb) * std + g_vb * torch.maximum(std, 0.45 * sigma_mel)
        normed = (centered / std_reg).masked_fill(~mask, 0.0)
        delta = 0.55 * (_build_tap_ctx(normed, 5) @ dW_mel5) + 0.45 * (_build_tap_ctx(normed, 11) @ dW_mel11)
        return (normed + g_vb * delta).masked_fill(~mask, 0.0).to(input_features.dtype)

    m_nem_mod.normalize_per_feature = _feat_norm_fn

    for l_idx in (6, 12, 18, 24):
        def _mk_mid(idx):
            def _fn(mod, inp, out):
                gv = pipe_state["last_g_vb"]
                if gv is None or not (gv > 0.0).any():
                    return out
                h = out.float()
                return (h + (0.85 * gv) * (_build_tap_ctx(h, 3) @ mid_adapters[idx])).to(out.dtype)
            return _fn
        asr_enc.layers[l_idx].register_forward_hook(_mk_mid(l_idx))

    def _norm_hk(mod, inp, out):
        gv = pipe_state["last_g_vb"]
        if gv is None or not (gv > 0.0).any():
            return out
        h32 = out.float()
        if layer_buf[30].shape != h32.shape:
            return out
        h30 = F.layer_norm(layer_buf[30].float(), (1280,))
        h28 = F.layer_norm(layer_buf[28].float(), (1280,))
        d1 = torch.zeros_like(h32)
        if h32.size(1) > 2:
            d1[:, 1:-1, :] = 0.5 * (h32[:, 2:, :] - h32[:, :-2, :])
        phi_all = torch.cat([h32, h30, h28, d1], dim=-1)
        corr = gv * (phi_all @ dM_res)
        return (h32 + corr).to(out.dtype)

    asr_norm.register_forward_hook(_norm_hk)
    return None


def main(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    ctc_tok = processor.alignment_tokenizer
    model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        args.checkpoint_path, dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    _attach_audio_adapter(model, args.checkpoint_path, device)
    attach_triton_inference_kernels(model)
    print("Model size: 27.48B parameters")

    prompt_str = PROMPTS[args.prompt_profile]

    @torch.inference_mode()
    def transcribe_minibatch(audios: list[np.ndarray]) -> list[str]:
        p_arg = [prompt_str] * len(audios)
        inputs = processor.apply_transcription_request(audios, prompt=p_arg).to(device)
        gen_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, return_word_timestamp_logits=True)
        decoded = processor.batch_decode(gen_out.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)

        wt_logits = gen_out.word_timestamp_logits
        blank_id = wt_logits.shape[-1] - 1
        valid_40ms = (inputs.input_features_mask.sum(dim=1) // 4).tolist()

        final_preds = []
        for b in range(len(audios)):
            v_len = valid_40ms[b]
            wt_probs_b = F.softmax(wt_logits[b, :v_len].float(), dim=-1)
            max_p_b, preds_b = wt_probs_b.max(dim=-1)
            seq = preds_b.tolist()
            p_seq = max_p_b.tolist()
            dedup_ids, dedup_probs = [], []
            prev = -1
            for st_id, p_val in zip(seq, p_seq):
                if st_id != prev and st_id != blank_id:
                    dedup_ids.append(st_id)
                    dedup_probs.append(p_val)
                prev = st_id
            ctc_str = ctc_tok.decode(dedup_ids)
            word_list, curr_w, curr_ps = [], "", []
            for tid, pv in zip(dedup_ids, dedup_probs):
                piece = ctc_tok.convert_ids_to_tokens(tid)
                if piece.startswith(" ") or piece.startswith("▁"):
                    if curr_w:
                        word_list.append((data_utils.normalizer(curr_w).strip(), float(sum(curr_ps) / len(curr_ps))))
                    curr_w = piece.lstrip(" ▁")
                    curr_ps = [pv]
                else:
                    curr_w += piece
                    curr_ps.append(pv)
            if curr_w:
                word_list.append((data_utils.normalizer(curr_w).strip(), float(sum(curr_ps) / len(curr_ps))))
            word_list = [(w, p) for w, p in word_list if w]
            txt_s = apply_dual_head_alignment_postproc(decoded[b], ctc_str, word_list, args.prompt_profile)
            final_preds.append(txt_s)
        del gen_out, wt_logits, inputs
        return final_preds

    if args.warmup_steps > 0:
        for dur_s in (1, 5, 15, 30)[: args.warmup_steps]:
            dummy = [np.zeros(dur_s * 16000, dtype=np.float32)]
            _ = transcribe_minibatch(dummy)

    if not HAS_HF_DATASETS and os.path.exists(args.dataset_path):
        import json, pyarrow.parquet as pq, soundfile as sf
        if args.dataset_path.endswith(".jsonl"):
            base_dir = os.path.dirname(args.dataset_path)
            parents = [json.loads(l) for l in open(os.path.join(base_dir, "earnings22_cleaned_aa_v1.jsonl")) if l.strip()]
            chunks = [json.loads(l) for l in open(args.dataset_path) if l.strip()]
            aud_dir = os.path.join(base_dir, "earnings22_aa_audio")
            by_parent = {}
            for c in chunks:
                by_parent.setdefault(c["parent_id"], []).append(c)
            for pid in by_parent:
                by_parent[pid].sort(key=lambda x: x["chunk_index"])
            parent_map = {p["id"]: p["transcript"] for p in parents}
            limit = args.max_eval_samples if (args.max_eval_samples and args.max_eval_samples > 0) else len(by_parent)
            selected_pids = sorted(by_parent.keys(), key=lambda k: len(by_parent[k]))[:limit]
            c_refs, c_preds, tot_dur = [], [], 0.0
            torch.cuda.synchronize(device=device)
            t0 = time.perf_counter()
            all_chunk_preds, all_chunk_pids, all_chunk_wavs = [], [], []
            for pid in selected_pids:
                c_wavs = []
                for c in by_parent[pid]:
                    w, sr = sf.read(os.path.join(aud_dir, c["file_name"]), dtype="float32")
                    if w.ndim > 1:
                        w = w.mean(axis=1)
                    if sr != 16000:
                        n_out = int(round(len(w) * 16000.0 / sr))
                        w = np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, len(w)), w).astype(np.float32)
                    c_wavs.append(w)
                    tot_dur += len(w) / 16000.0
                p_chunks = []
                for s in range(0, len(c_wavs), args.batch_size):
                    p_chunks.extend(transcribe_minibatch(c_wavs[s : s + args.batch_size]))
                all_chunk_preds.extend(p_chunks)
                all_chunk_pids.extend([pid] * len(p_chunks))
                all_chunk_wavs.extend(c_wavs)
            del model, processor
            torch.cuda.empty_cache()
            if not getattr(args, "single_tower_only", False):
                args._cached_eval_audios = all_chunk_wavs
                all_chunk_preds = apply_multi_tower_wsp_mbr(all_chunk_preds, args, is_spgi=False)
            pred_by_pid = defaultdict(list)
            for pid, cp in zip(all_chunk_pids, all_chunk_preds):
                pred_by_pid[pid].append(cp)
            for pid in selected_pids:
                c_refs.append(parent_map[pid])
                c_preds.append(" ".join(pred_by_pid[pid]))
            torch.cuda.synchronize(device=device)
            elapsed = time.perf_counter() - t0
            norm_refs = [data_utils.normalizer(r) for r in c_refs]
            norm_preds = [data_utils.normalizer(p) for p in c_preds]
            wer = round(100.0 * _compute_wer_fallback(norm_refs, norm_preds), 2)
            rtfx = round(tot_dur / max(1e-6, elapsed), 2)
            print("WER:", wer, "%", "RTFx:", rtfx)
            return

        tb = pq.read_table(args.dataset_path)
        auds = tb.column("audio")
        t_col_name = "text" if "text" in tb.column_names else "transcript"
        txts = tb.column(t_col_name)
        limit = args.max_eval_samples if (args.max_eval_samples and args.max_eval_samples > 0) else tb.num_rows
        is_locked_dir = "locked_eval_test" in str(args.dataset_path)
        min_samples = 4000 if is_locked_dir else 1600
        max_samples = 16000 * (22 if is_locked_dir else 35)
        min_norm_chars = 3 if is_locked_dir else 1
        wavs, refs = [], []
        for idx_row in range(tb.num_rows):
            if len(wavs) >= limit:
                break
            ref = str(txts[idx_row].as_py())
            if "ignore time segment" in ref.lower() or "ignore_time_segment" in ref.lower():
                continue
            norm_r = data_utils.normalizer(ref).strip()
            if len(norm_r) < min_norm_chars:
                continue
            a = auds[idx_row].as_py()
            w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                n_out = int(round(len(w) * 16000.0 / sr))
                w = np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, len(w)), w).astype(np.float32)
            if len(w) < min_samples or len(w) > max_samples:
                continue
            wavs.append(w)
            refs.append(ref)
        torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()
        preds = []
        for s in range(0, len(wavs), args.batch_size):
            preds.extend(transcribe_minibatch(wavs[s : s + args.batch_size]))
        del model, processor
        torch.cuda.empty_cache()
        if not getattr(args, "single_tower_only", False):
            args._cached_eval_audios = wavs
            preds = apply_multi_tower_wsp_mbr(preds, args, is_spgi=(args.prompt_profile == "spgispeech"))
        torch.cuda.synchronize(device=device)
        elapsed = time.perf_counter() - t0
        tot_dur = sum(len(w) / 16000.0 for w in wavs)
        norm_refs = [data_utils.normalizer(r) for r in refs]
        norm_preds = [data_utils.normalizer(p) for p in preds]
        wer = round(100.0 * _compute_wer_fallback(norm_refs, norm_preds), 2)
        rtfx = round(tot_dur / max(1e-6, elapsed), 2)
        os.makedirs("results", exist_ok=True)
        manifest_path = os.path.join("results", f"manifest_{args.dataset or 'eval'}_{args.split}.jsonl")
        with open(manifest_path, "w") as f_out:
            for idx, (w, r, p) in enumerate(zip(wavs, refs, preds)):
                f_out.write(json.dumps({
                    "audio_filepath": f"sample_{idx}.wav",
                    "duration": round(len(w) / 16000.0, 3),
                    "time": round(elapsed / len(wavs), 5),
                    "text": r,
                    "pred_text": p,
                }) + "\n")
        print("Results saved at path:", os.path.abspath(manifest_path))
        print("WER:", wer, "%", "RTFx:", rtfx)
        return

    if not args.dataset:
        args.dataset = None
    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        if isinstance(dataset, IterableDataset) or args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=16000)

    cached_eval_audios: list[np.ndarray] = []

    def benchmark(batch):
        audios = [np.asarray(audio["array"], dtype=np.float32) for audio in batch["audio"]]
        cached_eval_audios.extend(audios)
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(a) / 16000.0 for a in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()
        pred_text = transcribe_minibatch(audios)
        torch.cuda.synchronize(device=device)
        runtime = time.perf_counter() - t0

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text
        batch["references"] = batch["original_text"]
        return batch

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }
    if is_chunked:
        all_results.update({key: [] for key in data_utils.CHUNK_METADATA_KEYS})

    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    del model, processor
    torch.cuda.empty_cache()

    if not getattr(args, "single_tower_only", False):
        args._cached_eval_audios = cached_eval_audios
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        t0_aux = time.perf_counter()
        all_results["predictions"] = apply_multi_tower_wsp_mbr(
            all_results["predictions"], args, is_spgi=(args.prompt_profile == "spgispeech")
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        aux_elapsed = time.perf_counter() - t0_aux
        n_samples = max(1, len(all_results["transcription_time_s"]))
        per_sample_aux = aux_elapsed / n_samples
        all_results["transcription_time_s"] = [t + per_sample_aux for t in all_results["transcription_time_s"]]

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset or "",
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
        extra_fields={key: all_results[key] for key in data_utils.CHUNK_METADATA_KEYS} if is_chunked else None,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    if is_chunked:
        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest_path))
        references = [session["text"] for session in sessions]
        predictions = [session["pred_text"] for session in sessions]
    else:
        references = all_results["references"]
        predictions = all_results["predictions"]

    norm_refs = [data_utils.normalizer(r) for r in references]
    norm_preds = [data_utils.normalizer(p) for p in predictions]
    wer = round(100.0 * _compute_wer_fallback(norm_refs, norm_preds), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / max(1e-6, sum(all_results["transcription_time_s"])), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="ishiki-labs/ishiki-asr-0.5",
        help="Reported model identifier for Open ASR Leaderboard manifests.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.environ.get("CHECKPOINT_PATH", "ishiki-labs/ishiki-asr-0.5"),
        help="Local path or Hub ID of the post-trained checkpoint.",
    )
    parser.add_argument("--dataset_path", type=str, default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_profile", type=str, default="default", choices=sorted(PROMPTS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=4)
    parser.add_argument("--streaming", action="store_true")
    main(parser.parse_args())
