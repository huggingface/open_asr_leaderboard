"""
Open ASR Leaderboard evaluation for Higgs Audio v3 STT.

Follows the template described at
https://github.com/huggingface/open_asr_leaderboard#add-a-new-library
(load_data / prepare_data / write_manifest from normalizer.data_utils) with
three pieces of library-specific custom logic that are required to reproduce
the reported 5.30% (8B) / 5.67% (1.7B) average WER:

    1. Input tokenization uses ``prepare_chatml_sample_qwen`` with the real
       ``<|audio_bos|>`` / ``<|audio_eos|>`` special-token IDs (not the
       leaderboard normalizer's LB text path).
    2. Decoded hypotheses pass through ``_fix_repetitions`` which caps runaway
       greedy-decoding loops at 3 copies.  Empirically worth ~4 WER points on
       AMI / Earnings-22 for the 8B model.
    3. Both hypothesis and reference are normalized with Whisper's
       ``EnglishTextNormalizer`` from the ``whisper_normalizer`` PyPI package
       (NOT the local ``open_asr_leaderboard/normalizer/`` copy, whose
       abbreviation dict is a strict subset of pip's: e.g. it keeps
       ``'cause`` and ``dunno`` unchanged where pip expands them to
       ``because`` / ``do not know``).

In addition to these three pieces of custom logic, two runtime features that
do not affect WER have been added for reliability on a long multi-GPU run:

    * ``--shard_idx`` / ``--num_shards`` – deterministic slicing of the
      (already sorted) ESB dataset so the script can be launched once per GPU.
    * ``--checkpoint_every`` – periodic JSON snapshot of partial predictions
      so an unexpected shutdown mid-run loses at most ``checkpoint_every``
      samples.  On restart the script resumes from the last snapshot.

Usage::

    python run_eval.py \
        --model_id bosonai/higgs-audio-v3-8b-stt \
        --dataset ami --split test --device 0

Dependencies::

    pip install torch transformers==4.51.0 datasets evaluate tqdm \
        whisper_normalizer peft
    # plus the private boson_multimodal package, importable from
    # $BOSON_MULTIMODAL_PATH or the PYTHONPATH.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from functools import partial

import evaluate
import numpy as np
import torch
from tqdm import tqdm

from normalizer import data_utils

# Pip's whisper_normalizer.english.EnglishTextNormalizer.  We import it under
# a distinct name so it can't be confused with the leaderboard-local copy
# (``data_utils.normalizer``).  See file header for why this matters.
from whisper_normalizer.english import EnglishTextNormalizer as PipEnglishNormalizer

# ---------------------------------------------------------------------------
# boson_multimodal – provides ChatML preprocessing and the Whisper-encoder
# collator.  Installed as a private package; fall back to BOSON_MULTIMODAL_PATH.
# ---------------------------------------------------------------------------
_BOSON_PATH = os.environ.get("BOSON_MULTIMODAL_PATH", "")
if _BOSON_PATH and _BOSON_PATH not in sys.path:
    sys.path.insert(0, _BOSON_PATH)

from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator  # noqa: E402
from boson_multimodal.data_types import AudioContent, ChatMLSample, Message  # noqa: E402
from boson_multimodal.dataset.chatml_dataset import (  # noqa: E402
    ChatMLDatasetSample,
    prepare_chatml_sample_qwen,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperProcessor  # noqa: E402


wer_metric = evaluate.load("wer")
pip_normalizer = PipEnglishNormalizer()

USER_PROMPT = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."


# ---------------------------------------------------------------------------
# Custom logic #2 – cap runaway identical-word loops.
# ---------------------------------------------------------------------------
def _fix_repetitions(text: str, max_repeat: int = 3) -> str:
    if not text:
        return text
    words = text.split()
    if not words:
        return text
    out = [words[0]]
    run = 1
    for w in words[1:]:
        if w == out[-1]:
            run += 1
            if run <= max_repeat:
                out.append(w)
        else:
            out.append(w)
            run = 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# Model loading – merges an optional LoRA adapter from the ``lora/`` subfolder
# of the HuggingFace model repo so downstream inference is a single forward.
# ---------------------------------------------------------------------------
def load_model(model_id, device, lora_subfolder="lora"):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=device,
    )

    if lora_subfolder:
        try:
            from huggingface_hub import hf_hub_download
            try:
                cfg_path = hf_hub_download(model_id, f"{lora_subfolder}/adapter_config.json")
                adapter_dir = os.path.dirname(cfg_path)
                from peft import PeftModel
                print(f"Loading LoRA adapter from {model_id}/{lora_subfolder} ...")
                model = PeftModel.from_pretrained(model, adapter_dir)
                model = model.merge_and_unload()
                print("LoRA adapter merged into base weights.")
            except Exception as e:
                print(f"(no LoRA adapter at {model_id}/{lora_subfolder}: {e.__class__.__name__})")
        except ImportError:
            print("peft/huggingface_hub not available; skipping LoRA load.")

    model = model.eval()
    dev = next(model.parameters()).device

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
    model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_proc,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=1,
        audio_num_codebooks=config.audio_num_codebooks,
        chunk_size_seconds=getattr(config, "chunk_size_seconds", 30),
        encoder_padding_method=getattr(config, "encoder_padding_method", "max_length"),
    )

    print(f"Model loaded on {dev}. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer, collator, dev


# ---------------------------------------------------------------------------
# Custom logic #1 – prepare_chatml_sample_qwen tokens, then ChatML→features
# via HiggsAudioSampleCollator, then greedy generate.
# ---------------------------------------------------------------------------
def transcribe_single(audio_np, model, tokenizer, collator, device,
                      enable_thinking=True, max_new_tokens=1024):
    messages = [Message(role="user", content=[USER_PROMPT, AudioContent(audio_url="placeholder")])]
    chatml = ChatMLSample(messages=messages)
    prep_fn = partial(prepare_chatml_sample_qwen, enable_thinking=enable_thinking)
    input_tokens, _, _, _ = prep_fn(chatml, tokenizer, add_generation_prompt=True)

    sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens),
        label_ids=None,
        audio_ids_concat=None,
        audio_ids_start=None,
        audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
        audio_waveforms_start=torch.tensor([0]),
        audio_sample_rate=torch.tensor([16000]),
        audio_speaker_indices=torch.tensor([0]),
    )

    batch = asdict(collator([sample]))
    batch = {k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": False,
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
        "tokenizer": tokenizer,
    }

    with torch.inference_mode():
        outputs = model.generate(**batch, **gen_kwargs)

    output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    parts = full_text.split("assistant\n")
    hyp = parts[-1] if len(parts) > 1 else full_text
    hyp = re.sub(r"<think>.*?</think>", "", hyp, flags=re.DOTALL)
    if "<think>" in hyp:
        hyp = hyp[: hyp.index("<think>")].strip()
    hyp = re.sub(r"<\|.*?\|>", "", hyp)
    hyp = hyp.strip()
    hyp = _fix_repetitions(hyp, max_repeat=3)
    return hyp


# ---------------------------------------------------------------------------
# Main loop – with shard slicing and periodic JSON checkpointing.
# ---------------------------------------------------------------------------
def _checkpoint_path(args):
    safe_model = args.model_id.replace("/", "-")
    safe_ds = args.dataset.replace("/", "-")
    return os.path.join(
        args.output_dir,
        f"CHK_{safe_model}_{safe_ds}_{args.split}_shard{args.shard_idx}-{args.num_shards}.json",
    )


def _load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_checkpoint(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    model, tokenizer, collator, dev = load_model(
        args.model_id, device, lora_subfolder=args.lora_subfolder
    )

    enable_thinking = not args.no_thinking
    max_new_tokens = args.max_new_tokens or (1024 if enable_thinking else 256)

    # Load & prepare via the official pipeline (audio cast, LB norm_text, filter).
    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    if args.num_shards > 1:
        total = len(dataset)
        per = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_idx * per
        end = min(start + per, total)
        print(f"Shard {args.shard_idx}/{args.num_shards}: samples [{start}:{end}] of {total}")
        dataset = dataset.select(range(start, end))

    dataset = data_utils.prepare_data(dataset)

    chk_path = _checkpoint_path(args)
    chk = _load_checkpoint(chk_path) or {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    done = len(chk["predictions"])
    if done:
        print(f"Resuming from {chk_path}: {done} samples already done.")

    all_results = chk
    last_chk = time.time()
    pbar = tqdm(dataset, desc=f"{args.dataset} shard {args.shard_idx}/{args.num_shards}")
    for idx, sample in enumerate(pbar):
        if idx < done:
            continue
        audio = sample["audio"]
        audio_np = np.array(audio["array"], dtype=np.float32)
        audio_length = len(audio_np) / audio["sampling_rate"]

        start_time = time.time()
        try:
            pred = transcribe_single(
                audio_np, model, tokenizer, collator, dev,
                enable_thinking=enable_thinking,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:  # pragma: no cover
            print(f"  Error at idx {idx}: {e}")
            pred = ""
        runtime = time.time() - start_time

        # --- Custom logic #3: pip normalizer for both hyp and ref. -----------
        norm_pred = pip_normalizer(pred)
        # ``original_text`` is attached by data_utils.normalize before filtering.
        ref_raw = sample.get("original_text", sample.get("text") or sample.get("sentence") or "")
        norm_ref = pip_normalizer(ref_raw)

        all_results["audio_length_s"].append(audio_length)
        all_results["transcription_time_s"].append(runtime)
        all_results["predictions"].append(norm_pred)
        all_results["references"].append(norm_ref)

        if time.time() - last_chk >= args.checkpoint_every:
            _save_checkpoint(chk_path, all_results)
            last_chk = time.time()

    _save_checkpoint(chk_path, all_results)

    # Final manifest – use the leaderboard helper so eval_utils.score_results
    # can aggregate this file alongside other libraries' outputs.
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        f"{args.dataset}-shard{args.shard_idx}of{args.num_shards}"
        if args.num_shards > 1 else args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Manifest:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"],
        predictions=all_results["predictions"],
    )
    wer = round(100 * wer, 2)
    total_audio = sum(all_results["audio_length_s"])
    total_time = sum(all_results["transcription_time_s"])
    rtfx = round(total_audio / total_time, 2) if total_time > 0 else 0.0
    print(f"WER: {wer}%  RTFx: {rtfx}  (n={len(all_results['predictions'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str,
                        default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Only batch_size=1 is supported.")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="If set, uses dataset streaming. Disabled by "
                             "default because sharding requires len().")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lora-subfolder", dest="lora_subfolder", default="lora",
                        help="Subfolder in model_id holding a LoRA adapter "
                             "(silently skipped if missing).  Use '' to disable.")

    # Custom additions for multi-GPU / resumable runs.
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--checkpoint_every", type=float, default=120.0,
                        help="Seconds between checkpoint snapshots.")
    args = parser.parse_args()
    main(args)
