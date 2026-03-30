import argparse
import os
from pathlib import Path
import sys
import time

import evaluate
import torch
from tqdm import tqdm


from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from normalizer import data_utils


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def load_model(model_id, device):
    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, revision="refs/pr/6")
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        revision="refs/pr/6"
    ).to(device)
    model.eval()
    return processor, model


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    processor, model = load_model(args.model_id, device)

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        sample_rates = [audio["sampling_rate"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [
            len(audio) / sample_rate for audio, sample_rate in zip(audios, sample_rates)
        ]

        start_time = time.time()

        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            language=args.language,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(device=device, dtype=model.dtype)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
            pred_text = processor.decode(
                outputs,
                skip_special_tokens=True,
                language=args.language,
                audio_chunk_index=audio_chunk_index,
            )

        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(
                range(min(num_warmup_samples, len(warmup_dataset)))
            )
        warmup_dataset = iter(
            warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True)
        )

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with Transformers.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name from the benchmark dataset repository.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Required ISO 639-1 language code for Cohere Transcribe.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the model on. -1 for CPU, 0 for the first GPU, and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of audio samples to process per benchmark batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate. Use a small number for testing.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Download the whole dataset instead of streaming it.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before timed evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--no-punctuation",
        dest="punctuation",
        action="store_false",
        help="Disable punctuation in the model output.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False, punctuation=True)

    main(args)
