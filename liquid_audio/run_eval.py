"""Evaluate the unmodified Liquid Audio implementation, not native Transformers Lfm2Audio.

The upstream generator indexes batch item zero and supports one utterance per call.
--batch_size controls dataset batching; inference remains sequential with effective
model batch size one. Timings include prompt construction, the frontend, generation,
and text decoding, with CUDA synchronization and untimed warmup.
"""

import argparse
import importlib.metadata
import itertools
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from normalizer import data_utils
from normalizer.eval_utils import score_results

CHECKPOINT = "LiquidAI/LFM2.5-Audio-1.5B"
CHECKPOINT_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
REFERENCE_REVISION = "19e65845923a7f136442c95137884ec61eb386aa"
TRANSFORMERS_REVISION = "89a727319005f512aff3714c3838c8c37630b4fa"
LEADERBOARD_REVISION = "48e6f0d75ad7dad68c962112356fec44776f3bbb"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=CHECKPOINT)
    parser.add_argument("--revision", default=CHECKPOINT_REVISION)
    parser.add_argument("--dataset_path", default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--split", default="test.clean")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dataset batch size; the original generator runs each utterance separately.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    # Materialize datasets before timing. Partial remote Parquet iterators can
    # also crash during interpreter shutdown in the pinned Arrow environment.
    parser.set_defaults(streaming=False)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.warmup_steps < 0:
        parser.error(
            "batch_size/max_new_tokens must be positive and warmup_steps nonnegative"
        )
    if args.max_eval_samples == 0 or args.max_eval_samples < -1:
        parser.error("max_eval_samples must be -1 (all) or positive")
    if args.dataset == "":
        args.dataset = None
    return args


@torch.inference_mode()
def main(args):
    torch.set_float32_matmul_precision("highest")
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    checkpoint = Path(snapshot_download(args.model_id, revision=args.revision))
    processor = LFM2AudioProcessor.from_pretrained(checkpoint, device=device).eval()
    model = LFM2AudioModel.from_pretrained(
        checkpoint, dtype=dtype, device=device
    ).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Model size: {parameter_count / 1e9:.2f}B parameters ({parameter_count})",
        flush=True,
    )
    print("Original generator: effective model batch size = 1", flush=True)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def transcribe(audio):
        synchronize()
        start = time.perf_counter()
        chat = ChatState(processor, dtype=dtype)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        chat.new_turn("user")
        waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
        if waveform.ndim != 1:
            raise ValueError("Expected mono audio after dataset preparation")
        chat.add_audio(waveform.unsqueeze(0), audio["sampling_rate"])
        chat.end_turn()
        chat.new_turn("assistant")
        tokens = []
        for event in model.generate_sequential(
            **chat, max_new_tokens=args.max_new_tokens, text_top_k=1, audio_top_k=1
        ):
            if event.numel() != 1:
                raise RuntimeError("Original model generated audio for an ASR request")
            tokens.append(event.item())
        prediction = processor.text.decode(tokens, skip_special_tokens=True)
        synchronize()
        elapsed = time.perf_counter() - start
        truncated = len(tokens) == args.max_new_tokens and tokens[-1] != 7
        return prediction, elapsed, truncated

    dataset = data_utils.prepare_data(data_utils.load_data(args))
    # Use an independent iterator: warmup must never consume evaluation examples.
    for sample in itertools.islice(iter(dataset), args.warmup_steps):
        transcribe(sample["audio"])
    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    fields = [
        "references",
        "predictions",
        "audio_length_s",
        "transcription_time_s",
        "audio_filepath",
        "generation_truncated",
    ]
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    if is_chunked:
        fields += data_utils.CHUNK_METADATA_KEYS
    results = {key: [] for key in fields}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write one progress record per utterance, so a failed run retains diagnostics.
    # Only completed runs write a leaderboard manifest under results/.
    # Keep diagnostics outside the scorer's *.jsonl manifest search.
    progress_path = output_dir / "progress.ndjson"
    with progress_path.open("w") as progress:
        with tqdm(
            total=None if args.streaming else len(dataset), desc="Utterances"
        ) as bar:
            for batch in dataset.iter(batch_size=args.batch_size):
                filepaths = data_utils.extract_audio_filepaths_from_batch(
                    batch, len(batch["audio"])
                )
                for index, audio in enumerate(batch["audio"]):
                    prediction, elapsed, truncated = transcribe(audio)
                    record = {
                        "references": batch["original_text"][index],
                        "predictions": prediction,
                        "audio_length_s": len(audio["array"]) / audio["sampling_rate"],
                        "transcription_time_s": elapsed,
                        "audio_filepath": filepaths[index],
                        "generation_truncated": truncated,
                    }
                    if is_chunked:
                        record.update(
                            {
                                key: batch[key][index]
                                for key in data_utils.CHUNK_METADATA_KEYS
                            }
                        )
                    for key in fields:
                        results[key].append(record[key])
                    progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.flush()
                    bar.update(1)
    if not results["references"]:
        raise ValueError("No nonempty-reference examples were evaluated")

    # The shared manifest writer uses ./results; keep all run artifacts together.
    previous_dir = Path.cwd()
    try:
        os.chdir(output_dir)
        extra_keys = ["generation_truncated"] + (
            data_utils.CHUNK_METADATA_KEYS if is_chunked else []
        )
        manifest = data_utils.write_manifest(
            results["references"],
            results["predictions"],
            args.model_id,
            args.dataset_path,
            args.dataset or "",
            args.split,
            audio_length=results["audio_length_s"],
            transcription_time=results["transcription_time_s"],
            audio_filepaths=results["audio_filepath"],
            extra_fields={key: results[key] for key in extra_keys},
        )
        manifest = Path(manifest).resolve()
    finally:
        os.chdir(previous_dir)
    metadata = {
        "arguments": vars(args),
        "checkpoint_revision": checkpoint.name,
        "reference_revision": REFERENCE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "leaderboard_revision": LEADERBOARD_REVISION,
        "effective_model_batch_size": 1,
        "parameter_count": parameter_count,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "dtype": str(dtype),
        "samples": len(results["references"]),
        "complete_split": args.max_eval_samples == -1 and "[" not in args.split,
        "truncated_generations": sum(results["generation_truncated"]),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "liquid-audio",
                "transformers",
                "torch",
                "torchaudio",
                "datasets",
                "librosa",
            )
        },
    }
    manifest.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Manifest:", manifest, flush=True)
    print("Truncated generations:", metadata["truncated_generations"], flush=True)
    score_results(str(manifest.parent), args.model_id)


if __name__ == "__main__":
    main(parse_args())
