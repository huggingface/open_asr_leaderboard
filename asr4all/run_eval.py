"""Open ASR Leaderboard runner for FUTO asr4all models (trust_remote_code CTC + PCEC).

Loads futo-org/asr4all-{s,m,l} with AutoModelForCTC(trust_remote_code=True), batches
length-sorted audio through the encoder (bf16 autocast, torch.compile, CUDA graphs, fused Triton
kernels from `asr4all_kernels`), runs the model's punctuation/casing/error-correction pass (PCEC)
on the batch, and writes RAW transcripts through normalizer.data_utils so the leaderboard's
normalizer scores them. RTFx = total audio / total transcription time.

Warmup is the first --warmup_steps batches. Shapes those batches do not cover are compiled
before timing from synthetic inputs, never from evaluation data:
  - every batch is padded to --batch_size rows (padding rows are zeros and are not scored);
  - the encoder compiles static, tuned kernels for the warmup batches' lengths, and one
    dynamic-length graph for every other length, built from two silent batches (one shorter
    and one longer than the 10.24 s full-context window);
  - the PCEC pass runs at one of four fixed widths (64/128/192/256 tokens); all four are
    compiled from dummy inputs.

One decoding configuration is used for every dataset: --trunk_op 256,64 decodes clips up to
10.24 s in a single window (full context); longer audio uses a sliding window (100 frames back,
64 ahead) computed block-sparse with FlexAttention.
"""

import argparse
import collections
import os
import time


def _cpu_quota() -> int:
    try:
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) / int(period)))
    except (OSError, ValueError):
        pass
    if os.environ.get("CPU_CORES", "").isdigit():
        return max(1, int(os.environ["CPU_CORES"]))
    return len(os.sched_getaffinity(0))


_N_CPU = _cpu_quota()
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, _N_CPU)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(8, _N_CPU)))
import numpy
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(min(4, _N_CPU))
from normalizer import data_utils
from normalizer.eval_utils import score_results

torch.set_float32_matmul_precision("high")
if "--bf16_reduce" in __import__("sys").argv[1:] and "--bf16_reduce 0" not in " ".join(
    __import__("sys").argv
):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
PAUSE_THRESHOLDS = (1, 2, 4, 8, 16, 32, 64, 128)


def collapse_pool_pause_batched(
    logits, hidden, *, blank: int, tier: int, pad_in: int, blackout: int, valid_frames
):
    B, T, _ = logits.shape
    cap, D = (int(tier), hidden.shape[-1])
    dev = logits.device
    ids = logits.argmax(-1)
    ar = torch.arange(T, device=dev)
    live_f = ar.view(1, T) < valid_frames.view(B, 1) if valid_frames is not None else None
    prev = torch.cat([ids.new_full((B, 1), -1), ids[:, :-1]], 1)
    em = (ids != prev) & (ids != blank)
    if live_f is not None:
        em = em & live_f
    n_max = int(torch.cumsum(em.long(), 1)[:, -1].max())
    H = next((s for s in (64, 128, 192, 256) if s >= min(n_max, cap) and s <= cap), cap)
    em = em & (torch.cumsum(em.long(), 1) <= H)
    cs = torch.cumsum(em.long(), 1)
    kk = torch.arange(1, H + 1, device=dev).view(1, H).expand(B, H)
    start = torch.searchsorted(cs, kk, right=False).clamp(max=T)
    n_tok = cs[:, -1].clamp(max=H)
    live = torch.arange(H, device=dev).view(1, H) < n_tok.view(B, 1)
    nxt = torch.cat([start[:, 1:], start.new_full((B, 1), T)], 1)
    cnt = torch.where(live, nxt - start, torch.zeros_like(start))
    start = start.clamp(max=T - 1)
    tok = ids.gather(1, start.clamp(max=T - 1))
    ids_out = torch.where(live, tok, torch.full_like(tok, int(pad_in)))
    P = torch.cumsum(hidden.float(), 1)
    hi = P.gather(1, (start + cnt - 1).clamp(0, T - 1).unsqueeze(-1).expand(B, H, D))
    lo = P.gather(1, (start - 1).clamp(min=0).unsqueeze(-1).expand(B, H, D))
    lo = lo * (start > 0).unsqueeze(-1).float()
    pooled = (hi - lo) / cnt.clamp(min=1).unsqueeze(-1).float()
    acoustic = (pooled * live.unsqueeze(-1).float()).to(hidden.dtype)
    nb = (ids != blank).long()
    if live_f is not None:
        nb = nb * live_f.long()
    C = torch.cumsum(nb, 1)
    prev_start = torch.cat([start.new_zeros(B, 1), start[:, :-1]], 1)
    hi_p = C.gather(1, (start - 1).clamp(0, T - 1)) * (start > 0).long()
    lo_p = C.gather(1, (prev_start - 1).clamp(min=0)) * (prev_start > 0).long()
    run = (start - prev_start - (hi_p - lo_p)).clamp(min=0)
    thr = torch.tensor(PAUSE_THRESHOLDS, dtype=run.dtype, device=dev)
    b = (run.unsqueeze(-1) >= thr.view(1, 1, -1)).long().sum(-1).clamp(max=8)
    pause = torch.where(live, b, torch.full_like(b, int(blackout)))
    return (ids_out, acoustic, pause, live.sum(-1))


class _HubCtrl:
    def __init__(self, cp):
        c = cp["ctrl"]
        self.blank = len(cp["vocab"]) + 5
        self.case = {c["cap"]: "cap", c["allcaps"]: "allcaps"}
        self.punct = {c["period"]: ".", c["comma"]: ",", c["question"]: "?"}


def _hub_render(seq, vocab, ctrl):
    words_out, pending = ([], "")
    for i in seq:
        if i in ctrl.case:
            pending = ctrl.case[i]
        elif i in ctrl.punct:
            if words_out:
                words_out[-1][2] = ctrl.punct[i]
        elif i < len(vocab) and vocab[i]:
            if vocab[i].startswith(" ") or not words_out:
                words_out.append([[vocab[i]], pending, ""])
                pending = ""
            else:
                words_out[-1][0].append(vocab[i])
    return words_out


def _hub_render_text(words_out):
    rendered, sentence_start = ([], True)
    for pieces, case, punct in words_out:
        w = "".join(pieces).strip()
        if not w:
            continue
        if case == "allcaps":
            w = w.upper()
        elif case == "cap" or sentence_start:
            w = w[:1].upper() + w[1:]
        sentence_start = punct in (".", "?")
        rendered.append(w + punct)
    return " ".join(rendered)


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    dtype = getattr(torch, args.dtype)
    on_cuda = device.startswith("cuda")
    amp = on_cuda and dtype != torch.float32
    from transformers import AutoModelForCTC

    model = AutoModelForCTC.from_pretrained(
        args.model_id, revision=args.revision, trust_remote_code=True
    )
    model = model.to(device).eval()
    encoder, vocab, blank = (model.encoder, model.config.vocab, model.config.blank_id)
    pcec, pc = (getattr(model, "pcec", None), None)
    if pcec is not None:
        cp = model.config.pcec
        pc = {
            "vocab": list(cp["vocab"]),
            "ctrl": _HubCtrl(cp),
            "tier": 256,
            "pad_in": int(cp["pad_in"]),
            "pause_blackout": int(cp["pause_blackout"]),
            "real_pause": bool(cp.get("real_pause", False)),
            "hub": True,
        }
        print(
            f"pcec (hub): {sum((p.numel() for p in pcec.parameters())) / 1000000.0:.2f}M params, real_pause={pc['real_pause']}"
        )
    if pcec is not None and args.torch_compile is not None:
        pcec = torch.compile(
            pcec,
            options={
                "max_autotune": args.torch_compile == "max-autotune",
                "triton.cudagraphs": False,
            },
            dynamic=False,
        )
        print(
            "torch.compile: pcec compiled (static; cudagraphs off — eager glue sits between regions)"
        )
    print(
        f"Model size: {sum((p.numel() for p in model.parameters())) / 1000000000.0:.4f}B parameters"
    )
    print(f"pcec: {('ENABLED (in the timed region)' if pcec is not None else 'off')}")
    adaptive = args.trunk_op == "batch"
    if adaptive:
        raise SystemExit("--trunk_op must be C,R (e.g. 256,64) for the banded op")
    encoder.force_op = tuple((int(x) for x in args.trunk_op.split(",")))
    op_desc = (
        f"batch-adaptive (T_batch, {encoder.ec.right_ctx})"
        if adaptive
        else f"banded-compiled {args.trunk_op}"
        if getattr(encoder, "force_masks", None)
        else encoder.force_op
    )
    print(f"[trunk-op] {op_desc}")
    if not args.no_bn_fold:
        try:
            from asr4all_kernels import fuse_conv_bn

            print(f"bn-fold: folded {fuse_conv_bn(model)} BatchNorms into adjacent conv/linear")
        except ImportError:
            print("bn-fold: asr4all_kernels not importable — skipped")
    if on_cuda and (not args.no_kernels):
        try:
            from dataclasses import replace
            from asr4all_kernels import enable_ldsa_kernel, enable_se_kernel

            ks = set(args.kernels.split(","))
            if ks & {"all", "se"}:
                enable_se_kernel(model)
            if ks & {"all", "ldsa"}:
                enable_ldsa_kernel(model)
            if ks & {"all", "fir"} - {"all"}:
                from asr4all_kernels import enable_fir_kernel

                enable_fir_kernel(model)
            encoder.ec = replace(encoder.ec, use_flex_attn=True)
            print(f"kernels: {args.kernels} (flex armed)")
        except ImportError:
            print("kernels: asr4all_kernels not importable — running eager (install it to enable)")

    class _Trunk(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, wav, ilen):
            feat, feat_lens = self.m.pre(wav, ilen)
            out = self.m.encoder(feat, feat_lens)
            return (self.m.ctc_head(out.hidden), out.hidden, feat_lens)

    trunk = _Trunk(model)
    if args.torch_compile is not None:
        torch._dynamo.config.cache_size_limit = 64
        import torch._inductor.config as _icfg

        opts = {"max_autotune": args.torch_compile == "max-autotune", "triton.cudagraphs": False}
        torch._dynamo.config.automatic_dynamic_shapes = False
        trunk = torch.compile(trunk, options=opts, dynamic=None)
        print(
            f"torch.compile: {opts}, dynamic={'time axis'} (trunk wrapper); inductor cudagraphs={getattr(_icfg.triton, 'cudagraphs', 'n/a')}"
        )
    render, render_text = (_hub_render, _hub_render_text)
    _n_slots = args.prefetch + 2
    _stage = {"bufs": [None] * _n_slots, "evs": [None] * _n_slots, "i": 0, "last": 0}
    _host = {"fill_s": 0.0, "prep_wait_s": 0.0, "drain_s": 0.0, "render_s": 0.0}

    def _rows(n):
        return max(n, args.batch_size)

    def _plens(lens):
        return list(lens) + [min(lens)] * (_rows(len(lens)) - len(lens))

    def _assemble(audios, lens):
        max_len = max(lens)
        if args.pad_multiple > 1:
            max_len = -(-max_len // args.pad_multiple) * args.pad_multiple
        B = _rows(len(audios))
        k = _stage["i"] = (_stage["i"] + 1) % _n_slots
        if _stage["evs"][k] is not None:
            _stage["evs"][k].synchronize()
            _stage["evs"][k] = None
        _stage["last"] = k
        need = B * max_len
        flat = _stage["bufs"][k]
        if flat is None or flat.numel() < need:
            _stage["bufs"][k] = flat = torch.zeros(
                max(B, args.batch_size) * max_len, dtype=torch.float32, pin_memory=on_cuda
            )
        buf = flat[:need].view(B, max_len)
        bn = buf.numpy()
        h0 = time.perf_counter()

        def _fill(rows):
            for j in rows:
                if j >= len(audios):
                    bn[j, :max_len] = 0.0
                    continue
                a = audios[j]
                n = len(a)
                bn[j, :n] = a
                bn[j, n:max_len] = 0.0

        _fill(range(B))
        _host["fill_s"] += time.perf_counter() - h0
        return buf

    _hyb = {"mark": False}

    def _dyn(inp):
        if _hyb["mark"]:
            torch._dynamo.mark_dynamic(inp, 1)
        return inp

    _pmeta: dict = {}

    def _pcec_run(ids, ac, pause):
        _pmeta.update(D=ac.shape[-1], ac=ac.dtype, ids=ids.dtype, pause=pause.dtype)
        z = pcec(ids, acoustic=ac, pause=pause)
        return z

    def _tier(logits):
        return pc["tier"]
        return min(pc["tier"] or logits.shape[1], logits.shape[1])

    def issue_prepared(inp, ilen, n):
        start = torch.cuda.Event(enable_timing=True) if on_cuda else None
        end = torch.cuda.Event(enable_timing=True) if on_cuda else None
        w0 = time.perf_counter()
        if start:
            start.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=amp):
            logits, hidden, feat_lens = trunk(_dyn(inp), ilen)
            vf = torch.div(feat_lens, encoder.ec.rate_reduce, rounding_mode="floor").clamp(
                max=logits.shape[1]
            )
            slots = ids_cpu = None
            if pcec is None:
                ids_cpu = logits.argmax(-1)
            if pcec is not None:
                tier = _tier(logits)
                ids, ac, pause, _n = collapse_pool_pause_batched(
                    logits,
                    hidden,
                    blank=blank,
                    tier=tier,
                    pad_in=pc["pad_in"],
                    blackout=pc["pause_blackout"],
                    valid_frames=vf,
                )
                if not pc["real_pause"]:
                    pause = torch.full_like(pause, int(pc["pause_blackout"]))
                z = _pcec_run(ids, ac, pause)
                slots = z.argmax(-1)
        if start:
            end.record()
        return {"slots": slots, "ids": ids_cpu, "vf": vf, "ev": (start, end), "w0": w0, "n": n}

    def issue_batch(audios, lens):
        batch = _assemble(audios, lens)
        inp = batch.to(device, non_blocking=True)
        ilen = torch.tensor(_plens(lens), device=device)
        start = torch.cuda.Event(enable_timing=True) if on_cuda else None
        end = torch.cuda.Event(enable_timing=True) if on_cuda else None
        w0 = time.perf_counter()
        if start:
            start.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=amp):
            logits, hidden, feat_lens = trunk(_dyn(inp), ilen)
            vf = torch.div(feat_lens, encoder.ec.rate_reduce, rounding_mode="floor").clamp(
                max=logits.shape[1]
            )
            slots = ids_cpu = None
            if pcec is None:
                ids_cpu = logits.argmax(-1)
            if pcec is not None:
                tier = _tier(logits)
                ids, ac, pause, _n = collapse_pool_pause_batched(
                    logits,
                    hidden,
                    blank=blank,
                    tier=tier,
                    pad_in=pc["pad_in"],
                    blackout=pc["pause_blackout"],
                    valid_frames=vf,
                )
                if not pc["real_pause"]:
                    pause = torch.full_like(pause, int(pc["pause_blackout"]))
                z = _pcec_run(ids, ac, pause)
                slots = z.argmax(-1)
        if start:
            end.record()
        return {
            "slots": slots,
            "ids": ids_cpu,
            "vf": vf,
            "ev": (start, end),
            "w0": w0,
            "n": len(audios),
        }

    def decode_async(h):
        if h["done"] is not None:
            h["done"].synchronize()
        if h.get("slots_host") is not None:
            return _render_slots(h)
        return _render_ids(h)

    def _render_slots(h):
        r0 = time.perf_counter()
        ctrl, pv = (pc["ctrl"], pc["vocab"])
        sm = h["slots_host"].numpy()
        prevm = numpy.concatenate([numpy.full((sm.shape[0], 1), -1, sm.dtype), sm[:, :-1]], 1)
        keep = (sm != prevm) & (sm != ctrl.blank)
        out = [render_text(render(sm[j][keep[j]].tolist(), pv, ctrl)) for j in range(h["n"])]
        _host["render_s"] += time.perf_counter() - r0
        return out

    def _render_ids(h):
        im = h["ids_host"].numpy()
        vfc = h["vf_host"].tolist()
        prev = numpy.concatenate([numpy.full((im.shape[0], 1), -1, im.dtype), im[:, :-1]], 1)
        em = (im != prev) & (im != blank) & (im < len(vocab))
        texts = []
        for j in range(h["n"]):
            row = im[j, : max(1, vfc[j])][em[j, : max(1, vfc[j])]]
            texts.append(" ".join("".join((vocab[i] for i in row)).split()))
        return texts

    def decode_result(h):
        slots, ids_cpu, vf = (h["slots"], h["ids"], h["vf"])
        if slots is not None:
            ctrl, pv = (pc["ctrl"], pc["vocab"])
            sm = slots.cpu().numpy()
            prevm = numpy.concatenate([numpy.full((sm.shape[0], 1), -1, sm.dtype), sm[:, :-1]], 1)
            keep = (sm != prevm) & (sm != ctrl.blank)
            texts = []
            for j in range(h["n"]):
                texts.append(render_text(render(sm[j][keep[j]].tolist(), pv, ctrl)))
        else:
            im = ids_cpu.cpu().numpy()
            vfc = vf.cpu().tolist()
            prev = numpy.concatenate([numpy.full((im.shape[0], 1), -1, im.dtype), im[:, :-1]], 1)
            em = (im != prev) & (im != blank) & (im < len(vocab))
            texts = []
            for j in range(h["n"]):
                row = im[j, : max(1, vfc[j])][em[j, : max(1, vfc[j])]]
                texts.append(" ".join("".join((vocab[i] for i in row)).split()))
        return texts

    def run_batch(audios, lens):
        h = issue_batch(audios, lens)
        texts = decode_result(h)
        if on_cuda:
            torch.cuda.synchronize(device)
        s, e = h["ev"]
        gpu_s = s.elapsed_time(e) / 1000.0 if s else 0.0
        return (texts, gpu_s, time.perf_counter() - h["w0"])

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        dataset = dataset.take(args.max_eval_samples)
    dataset = data_utils.prepare_data(dataset)
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)

    def _row(r):
        row = {
            "audio": numpy.asarray(r["audio"]["array"], dtype=numpy.float32),
            "ref": r.get("original_text") or r.get("text") or r["norm_text"],
        }
        if is_chunked:
            row["chunk"] = {k: r[k] for k in data_utils.CHUNK_METADATA_KEYS}
        return row

    m0 = time.perf_counter()
    n_dec = args.decode_threads if hasattr(dataset, "shard") and hasattr(dataset, "__len__") else 1
    n_dec = max(1, min(n_dec, len(dataset) // 64 if hasattr(dataset, "__len__") else 1))
    if n_dec > 1:
        from concurrent.futures import ThreadPoolExecutor

        shards = [dataset.shard(num_shards=n_dec, index=k, contiguous=True) for k in range(n_dec)]
        with ThreadPoolExecutor(max_workers=n_dec) as ex:
            parts = list(ex.map(lambda sh: [_row(r) for r in sh], shards))
        rows = [r for part in parts for r in part]
    else:
        rows = [_row(r) for r in dataset]
    del dataset
    print(f"decoded {len(rows)} rows in {time.perf_counter() - m0:.1f}s ({n_dec} threads)")
    order = sorted(
        range(len(rows)), key=lambda i: len(rows[i]["audio"]), reverse=args.sort == "desc"
    )

    def batches():
        for lo in range(0, len(order), args.batch_size):
            idxs = order[lo : lo + args.batch_size]
            audios = [rows[i]["audio"] for i in idxs]
            yield (idxs, audios, [len(a) for a in audios])

    if on_cuda:
        _max_len = max((len(r["audio"]) for r in rows))
        if args.pad_multiple > 1:
            _max_len = -(-_max_len // args.pad_multiple) * args.pad_multiple
        _rows_max = min(args.batch_size, len(rows))
        for k in range(_n_slots):
            _stage["bufs"][k] = torch.zeros(
                _rows_max * _max_len, dtype=torch.float32, pin_memory=True
            )

    def _shape_of(audios, lens):
        m = max(lens)
        if args.pad_multiple > 1:
            m = -(-m // args.pad_multiple) * args.pad_multiple
        return (_rows(len(audios)), m)

    seen_shapes = set()
    for bi, (_, audios, lens) in enumerate(batches()):
        if bi >= args.warmup_steps:
            break
        s = _shape_of(audios, lens)
        run_batch(audios, lens)
        if s not in seen_shapes:
            seen_shapes.add(s)
    _hyb["mark"] = True
    if on_cuda:
        c_ = int(args.trunk_op.split(",")[0])
        B = _rows(1)
        with torch.no_grad():
            feat, _ = model.pre(
                torch.zeros(1, 64000, device=device), torch.tensor([64000], device=device)
            )
            t_att = encoder.stem(encoder.input_proj(feat.transpose(1, 2))).shape[2]
        hop = 64000 // t_att
        step = args.pad_multiple if args.pad_multiple > 1 else hop

        def _unwarmed(samp, limit):
            while _shape_of([None] * B, [samp]) in seen_shapes and samp + step <= limit:
                samp += step
            return samp

        short = _unwarmed(min(step, c_ * hop), c_ * hop)
        long_ = _unwarmed(c_ * hop + hop, 1920000)
        for samp in (short, long_):
            if _shape_of([None] * B, [samp]) in seen_shapes:
                print(f"regime warmup: no unwarmed shape near {samp} samples (regime fully warmed)")
                continue
            z = numpy.zeros(samp, dtype=numpy.float32)
            run_batch([z] * B, [samp] * B)
        print(
            f"regime warmup: silent batches at {short} and {long_} samples (commit {c_} frames = {c_ * hop} samples)"
        )
    if pcec is not None and _pmeta and on_cuda:
        B = _rows(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=amp):
            for H in (64, 128, 192, 256):
                ids_d = torch.full((B, H), int(pc["pad_in"]), dtype=_pmeta["ids"], device=device)
                ac_d = torch.zeros(B, H, _pmeta["D"], dtype=_pmeta["ac"], device=device)
                pz_d = torch.full(
                    (B, H), int(pc["pause_blackout"]), dtype=_pmeta["pause"], device=device
                )
                _pcec_run(ids_d, ac_d, pz_d)
                _pcec_run(ids_d, ac_d, pz_d)
    print(f"warmup: covered {len(seen_shapes)} distinct padded shapes" + "" + "")
    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
        "row_idx": [],
    }
    tot_gpu = tot_wall = tot_audio = 0.0
    _host.update(fill_s=0.0, prep_wait_s=0.0, drain_s=0.0, render_s=0.0)
    from concurrent.futures import ThreadPoolExecutor

    def _prep(audios, lens):
        return _prep_inner(audios, lens)

    def _prep_inner(audios, lens):
        batch = _assemble(audios, lens)
        inp = batch.to(device, non_blocking=True)
        ilen = torch.tensor(_plens(lens), device=device)
        ev = None
        if on_cuda:
            done = torch.cuda.Event()
            done.record()
            _stage["evs"][_stage["last"]] = done
        return (inp, ilen, ev)

    texts_by_idx = {}
    prep_pool = ThreadPoolExecutor(max_workers=1)
    dec_pool = ThreadPoolExecutor(max_workers=1)
    if on_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    evs, dec_futs = ([], [])
    it = list(batches())
    futs = collections.deque((prep_pool.submit(_prep, b[1], b[2]) for b in it[: args.prefetch]))
    nxt = len(futs)
    for bi, (idxs, audios, lens) in enumerate(it):
        s0 = time.perf_counter()
        inp, ilen, ev = futs.popleft().result()
        _host["prep_wait_s"] += time.perf_counter() - s0
        if nxt < len(it):
            futs.append(prep_pool.submit(_prep, it[nxt][1], it[nxt][2]))
            nxt += 1
        if ev is not None:
            torch.cuda.current_stream(device).wait_event(ev)
        h = issue_prepared(inp, ilen, len(audios))
        h["inp"] = inp
        done = torch.cuda.Event() if on_cuda else None
        if h["slots"] is not None:
            h["slots_host"] = h["slots"].to("cpu", non_blocking=True)
        else:
            h["ids_host"] = h["ids"].to("cpu", non_blocking=True)
        h["vf_host"] = h["vf"].to("cpu", non_blocking=True)
        if done is not None:
            done.record()
        h["done"] = done
        evs.append(h["ev"])
        dec_futs.append((idxs, dec_pool.submit(decode_async, h)))
        tot_audio += sum(lens) / 16000
    s0 = time.perf_counter()
    for idxs, df in dec_futs:
        txt = df.result()
        texts_by_idx.update(zip(idxs, txt))
    if on_cuda:
        torch.cuda.synchronize(device)
    _host["drain_s"] += time.perf_counter() - s0
    tot_wall += time.perf_counter() - t0
    tot_gpu += sum((s.elapsed_time(e) / 1000.0 for s, e in evs if s is not None))
    prep_pool.shutdown(wait=False)
    dec_pool.shutdown(wait=False)
    for idxs, _audios3, lens in batches():
        for j, i in enumerate(idxs):
            results["references"].append(rows[i]["ref"])
            results["row_idx"].append(i)
            results["predictions"].append(texts_by_idx[i])
            results["audio_length_s"].append(lens[j] / 16000)
            results["transcription_time_s"].append(tot_wall * (lens[j] / 16000) / tot_audio)
    extra = None
    if is_chunked:
        keys = data_utils.CHUNK_METADATA_KEYS
        extra = {k: [rows[i]["chunk"][k] for i in results["row_idx"]] for k in keys}
    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
        extra_fields=extra,
    )
    print("results saved at path:", manifest_path)
    if is_chunked:
        import json as _json

        merged = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest_path))
        with open(manifest_path, "w") as f:
            for m in merged:
                f.write(_json.dumps(m) + "\n")
        print(f"chunk merge: {len(results['references'])} chunks -> {len(merged)} recordings")
    from pathlib import Path

    wer, rtfx = score_results(str(Path(manifest_path).parent), args.model_id)
    print(f"WER: {wer}  RTFx: {rtfx}")
    print(
        f"[rtfx] wall {tot_audio / tot_wall:.1f}   gpu {tot_audio / max(tot_gpu, 1e-09):.1f}   audio {tot_audio:.0f}s  wall {tot_wall:.1f}s  gpu {tot_gpu:.1f}s   host fill {_host['fill_s']:.1f}s ({1} thr) prep-wait {_host['prep_wait_s']:.1f}s drain {_host['drain_s']:.1f}s render {_host['render_s']:.1f}s   batch {args.batch_size}  op {op_desc}   {''}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True, help="HF repo id")
    p.add_argument("--revision", default="main", help="pin a tag/commit (required for TRC models)")
    p.add_argument("--dataset_path", default="hf-audio/open-asr-leaderboard")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16", help="bfloat16 | float16 | float32")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--torch_compile", default=None, help="e.g. 'default', 'max-autotune'")
    p.add_argument("--no_kernels", action="store_true", help="disable the fused SE/LDSA kernels")
    p.add_argument(
        "--kernels",
        default="all",
        help="comma set of fused kernels: all|se|ldsa|fir (fir is opt-in)",
    )
    p.add_argument("--no_bn_fold", action="store_true", help="disable eval BatchNorm folding")
    p.add_argument(
        "--trunk_op",
        default="batch",
        help="`batch` (offline-equivalent, per-batch commit) or `C,R`",
    )
    p.add_argument(
        "--sort",
        default="asc",
        choices=["asc", "desc"],
        help="duration sort direction (references use desc)",
    )
    p.add_argument(
        "--pad_multiple",
        type=int,
        default=1,
        help="round each batch's padded sample width up to this multiple (shape bucketing; 16000 = 1 s grid). 1 = off",
    )
    p.add_argument(
        "--decode_threads",
        type=int,
        default=8,
        help="threads decoding the dataset into memory before timing",
    )
    p.add_argument(
        "--prefetch", type=int, default=2, help="batches prepared ahead of the one being issued"
    )
    p.add_argument("--streaming", type=lambda x: x == "True", default=False)
    main(p.parse_args())
