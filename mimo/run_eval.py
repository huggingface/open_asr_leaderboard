"""
Evaluation runner for Xiaomi MiMo-V2.5-ASR on the Open ASR Leaderboard.

Loads MiMo locally via its custom MimoAudio class (not pip-installable —
must clone XiaomiMiMo/MiMo-V2.5-ASR and set MIMO_REPO_PATH).

Resumable: each completed sample is appended (and fsync'd) to the manifest
immediately, so re-running the same command picks up after the last
written row. Failures on individual samples log an empty pred and continue.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import evaluate
import torch
from tqdm import tqdm

from normalizer import data_utils

# MiMo's package layout: src.mimo_audio.mimo_audio.MimoAudio
# Caller must set MIMO_REPO_PATH env var so this import resolves.
if "MIMO_REPO_PATH" in os.environ:
    sys.path.insert(0, os.environ["MIMO_REPO_PATH"])

# Inject flash_attn shim BEFORE MiMo imports it, on hosts without flash-attn
# wheels (Windows native, exotic torch+CUDA combos). See flash_attn_shim.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import flash_attn  # noqa: F401  — real package present, use it
except ImportError:
    import flash_attn_shim
    sys.modules["flash_attn"] = flash_attn_shim

from src.mimo_audio.mimo_audio import MimoAudio  # noqa: E402


wer_metric = evaluate.load("wer")


def _manifest_path(model_id: str, dataset_path: str, dataset_name: str, split: str) -> str:
    """Mirror data_utils.write_manifest's naming so eval_utils.score_results can find it."""
    mid = model_id.replace("/", "-")
    dpath = dataset_path.replace("/", "-")
    dname = dataset_name.replace("/", "-")
    out = Path("./results") / f"MODEL_{mid}_DATASET_{dpath}_{dname}_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def _read_done_indices(manifest_path: str) -> int:
    """Count already-completed rows so we can resume from where we left off."""
    if not os.path.exists(manifest_path):
        return 0
    n = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main(args):
    print(f"Loading MiMo model from: {args.model_path}")
    print(f"  audio tokenizer:       {args.tokenizer_path}")
    print(f"  device:                cuda:{args.device}" if args.device >= 0 else "  device: cpu")
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    t0 = time.time()
    model = MimoAudio(
        model_path=args.model_path,
        mimo_audio_tokenizer_path=args.tokenizer_path,
        device=device,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling to first {args.max_eval_samples} samples (smoke test)")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    manifest = _manifest_path(args.model_id, args.dataset_path, args.dataset, args.split)
    done = _read_done_indices(manifest)
    if done > 0:
        print(f"Resuming: {done} samples already in {manifest}, skipping.")

    # Optional warmup — run a couple samples but don't write them, just to warm CUDA kernels.
    if args.warmup_steps and done == 0:
        print(f"Warming up on {args.warmup_steps} sample(s)...")
        warmup_iter = (dataset.take(args.warmup_steps) if args.streaming
                       else dataset.select(range(min(args.warmup_steps, len(dataset)))))
        for s in warmup_iter:
            try:
                wt = torch.from_numpy(np.asarray(s["audio"]["array"], dtype=np.float32))
                with torch.inference_mode():
                    _ = model.asr_sft(wt, audio_tag=args.audio_tag)
            except Exception as e:
                print(f"Warmup error (continuing): {type(e).__name__}: {e}")
                break

    f_out = open(manifest, "a", encoding="utf-8")
    try:
        idx = 0
        skipped = 0
        for sample in tqdm(dataset, desc="Inference"):
            if idx < done:
                idx += 1
                skipped += 1
                continue

            audio_arr = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            duration = float(len(audio_arr) / sr)
            # MiMo's torchcodec audio loader doesn't accept numpy arrays — convert.
            audio_tensor = torch.from_numpy(np.asarray(audio_arr, dtype=np.float32))

            t0 = time.time()
            try:
                with torch.inference_mode():
                    pred_text = model.asr_sft(audio_tensor, audio_tag=args.audio_tag)
            except Exception as e:
                print(f"[idx={idx}] inference error: {type(e).__name__}: {e}")
                pred_text = ""
            runtime = time.time() - t0

            pred_norm = data_utils.normalizer(pred_text or "")
            ref_norm = sample["norm_text"]

            row = {
                "audio_filepath": f"sample_{idx}",
                "duration": duration,
                "time": runtime,
                "text": ref_norm,
                "pred_text": pred_norm,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()
            os.fsync(f_out.fileno())
            idx += 1
        if skipped:
            print(f"Skipped {skipped} previously-completed samples on resume.")
    finally:
        f_out.close()

    # Final score
    refs, preds, durs, times = [], [], [], []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            refs.append(d["text"])
            preds.append(d["pred_text"])
            durs.append(d["duration"])
            times.append(d["time"])
    keep = [(r, p, d, t) for r, p, d, t in zip(refs, preds, durs, times) if r.strip()]
    refs, preds, durs, times = map(list, zip(*keep)) if keep else ([], [], [], [])

    if not refs:
        print("WARN: no scorable samples (manifest empty or all-empty refs).")
        return

    wer = round(100 * wer_metric.compute(references=refs, predictions=preds), 2)
    rtfx = round(sum(durs) / sum(times), 2) if sum(times) > 0 else float("nan")
    print(f"\nManifest: {os.path.abspath(manifest)}")
    print(f"WER: {wer:.2f}%   RTFx: {rtfx:.2f}   N={len(refs)}   audio={sum(durs)/3600:.2f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="XiaomiMiMo/MiMo-V2.5-ASR",
                        help="Used for manifest filename only.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local directory holding MiMo-V2.5-ASR weights.")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Local directory holding MiMo-Audio-Tokenizer.")
    parser.add_argument("--audio_tag", type=str, default="<english>",
                        help="MiMo language hint. <english> | <chinese> | '' (auto).")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index. -1 for CPU.")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit to first N samples (smoke test).")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.set_defaults(streaming=False)
    args = parser.parse_args()
    main(args)
