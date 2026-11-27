"""Open ASR Leaderboard evaluation (English short-form) for Ishiki-ASR-1.0 (`ishiki-labs/ishiki-asr-1.0`).

Loads the standalone single-file `IshikiASRForConditionalGeneration` checkpoint (`model.safetensors`, 13.31B BF16
parameters) through standard Hugging Face `AutoConfig`, `AutoProcessor`, and `AutoModelForSpeechSeq2Seq`, and runs
strictly per-utterance batched greedy `model.generate(**inputs)` + `processor.batch_decode(...)` with
`normalizer/data_utils.py` for dataset loading, raw un-normalized manifest writing, and scoring. Identical decoding
hyperparameters are used across every dataset with zero dataset-specific prompts, mode flags, or cross-utterance batch
dependencies.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

from normalizer import data_utils, voice_code_bench
from normalizer.eval_utils import align_manifests

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    torch_dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
    )
    if type(config) not in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        raise ValueError(f"Model config of type {type(config)} not recognized in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        revision=args.revision,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
    )
    sampling_rate = (
        processor.feature_extractor.sampling_rate
        if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None
        else 16_000
    )

    gen_kwargs = {"batch_size": args.batch_size}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    def benchmark(batch):
        audios = [np.asarray(audio["array"], dtype=np.float32) for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(a) / sampling_rate for a in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        if args.device >= 0:
            torch.cuda.synchronize(device=device)
        start_time = time.perf_counter()

        inputs = processor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        with torch.inference_mode():
            pred_ids = model.generate(**inputs, **gen_kwargs)
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        if args.device >= 0:
            torch.cuda.synchronize(device=device)
        runtime = time.perf_counter() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
        return batch

    is_vcb = voice_code_bench.is_voice_code_bench(args.dataset_path)

    if args.warmup_steps is not None and args.warmup_steps > 0:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset, sampling_rate=sampling_rate, normalize_text=not is_vcb)
        num_warmup_samples = min(args.warmup_steps * min(args.batch_size, 32), 64)
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_iter = iter(
            warmup_dataset.map(
                benchmark,
                batch_size=min(args.batch_size, 32),
                batched=True,
                remove_columns=["audio"],
            )
        )
        for _ in tqdm(warmup_iter, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate, normalize_text=not is_vcb)

    map_batch_size = max(args.batch_size, 4096)
    dataset = dataset.map(
        benchmark,
        batch_size=map_batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    metadata_keys = (
        voice_code_bench.MANIFEST_KEYS if is_vcb
        else data_utils.CHUNK_METADATA_KEYS if is_chunked else []
    )
    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }
    all_results.update({key: [] for key in metadata_keys})
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

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
        extra_fields={key: all_results[key] for key in metadata_keys} or None,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    if is_vcb:
        print("RTFx:", round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2))
        return

    if is_chunked:
        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest_path))
        references = [session["text"] for session in sessions]
        predictions = [session["pred_text"] for session in sessions]
    else:
        references = all_results["references"]
        predictions = all_results["predictions"]

    official_counts = align_manifests(
        [[{"text": r, "pred_text": p} for r, p in zip(references, predictions)]],
        num_workers=1,
    )[0]
    official_wer = round(100.0 * official_counts["err_rate"], 2)

    norm_refs = [data_utils.normalizer(r) for r in references]
    norm_preds = [data_utils.normalizer(p) for p in predictions]
    raw_wer = round(100.0 * wer_metric.compute(references=norm_refs, predictions=norm_preds), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print(f"Official WER (compound-aligned): {official_wer} % | Unaligned WER: {raw_wer} % | RTFx: {rtfx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="ishiki-labs/ishiki-asr-1.0",
        help="Model identifier on Hugging Face Hub or local path (`ishiki-labs/ishiki-asr-1.0`).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional immutable model commit SHA or tag on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/open-asr-leaderboard",
        help="Dataset path. By default, `hf-audio/open-asr-leaderboard`.",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional immutable dataset revision.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset config name (empty for single-config repos such as Earnings22-chunked or Monsoon).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (`test`, `test.clean`, `test.other`, etc.).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (`0` for `cuda:0`, `-1` for CPU).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="GPU inference batch size inside `model.generate`.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weight and compute dtype.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional cap on evaluated samples (e.g. `64` for smoke testing; `-1` or omitted for full split).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional override for maximum generated tokens.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of untimed warm-up steps before timed evaluation.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset lazily over the network (off by default for reproducible RTFx timings).",
    )
    args = parser.parse_args()
    main(args)
