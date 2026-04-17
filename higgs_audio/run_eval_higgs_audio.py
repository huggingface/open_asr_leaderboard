"""Evaluation script for Higgs Audio v3 models on ESB benchmark datasets.

No external dependencies beyond transformers + torch. The model bundles
its own audio preprocessing via trust_remote_code=True.

Usage:
    python run_eval_higgs_audio.py \
        --model_id bosonai/higgs-audio-v3-8b-stt \
        --dataset_path hf-audio/esb-datasets-test-only-sorted \
        --dataset ami --split test --device 0 --batch_size 4
"""

import argparse
import json
import os
import sys
import time
import runpy

import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoModel, AutoTokenizer
from normalizer import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def load_transcribe_fns(model_id):
    """Load the bundled transcribe + transcribe_batch functions from the model repo.

    Downloads all Python files needed by transcribe.py, then loads it via
    runpy with the download directory on sys.path so plain (non-relative)
    imports resolve to sibling files.
    """
    from transformers.utils import cached_file

    for filename in [
        "transcribe.py",
        "higgs_audio_collator.py",
        "modeling_higgs_audio_xcodec.py",
        "utils.py",
        "common.py",
        "configuration_higgs_audio.py",
    ]:
        cached_file(model_id, filename)

    path = cached_file(model_id, "transcribe.py")
    module_dir = os.path.dirname(path)

    sys.path.insert(0, module_dir)
    try:
        module_globals = runpy.run_path(path)
    finally:
        sys.path.pop(0)

    return module_globals["transcribe"], module_globals["transcribe_batch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Parallel transcription batch size (1 = single-sample loop)")
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    print(f"Loading model {args.model_id}...")
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.eval()

    # Required for generation stop conditions
    model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
    model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    transcribe, transcribe_batch = load_transcribe_fns(args.model_id)

    print(f"Loading dataset {args.dataset_path}/{args.dataset} ({args.split})...")
    dataset = load_dataset(args.dataset_path, args.dataset, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    print(f"Evaluating {len(dataset)} samples (batch_size={args.batch_size})...")

    predictions = []
    references = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    # Iterate in batches
    n = len(dataset)
    i = 0
    while i < n:
        batch_end = min(i + args.batch_size, n)
        batch_samples = [dataset[j] for j in range(i, batch_end)]

        audios = [np.array(s["audio"]["array"], dtype=np.float32) for s in batch_samples]
        refs = [s.get("norm_text", s.get("text", "")) for s in batch_samples]

        for a in audios:
            total_audio_duration += len(a) / 16000

        t0 = time.time()
        if args.batch_size > 1:
            preds = transcribe_batch(model, tokenizer, audios, sample_rates=16000)
        else:
            preds = [transcribe(model, tokenizer, audios[0])]
        total_inference_time += time.time() - t0

        for pred, ref in zip(preds, refs):
            predictions.append(normalizer(pred))
            references.append(normalizer(ref))

        i = batch_end
        if i % 100 == 0 or i == n:
            print(f"  {i}/{n} done", flush=True)

    from jiwer import wer
    wer_score = round(wer(references, predictions) * 100, 2)
    rtfx = round(total_audio_duration / total_inference_time, 2) if total_inference_time > 0 else 0

    print(f"\nResults for {args.model_id} on {args.dataset}:")
    print(f"  WER: {wer_score}%")
    print(f"  RTFx: {rtfx}")

    manifest = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "split": args.split,
        "batch_size": args.batch_size,
        "wer": wer_score,
        "rtfx": rtfx,
        "num_samples": len(dataset),
        "predictions": predictions,
        "references": references,
    }

    out_dir = os.path.join("results", args.model_id.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
