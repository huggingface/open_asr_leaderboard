import argparse
import os
from pathlib import Path
import sys
import time

import evaluate
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

TRANSFORMERS_COHERE_SRC = Path(__file__).resolve().parents[1] / "transformers_cohere" / "src"
if TRANSFORMERS_COHERE_SRC.is_dir():
    sys.path.insert(0, str(TRANSFORMERS_COHERE_SRC))

from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def load_model(model_id, device):
    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    return processor, model


def load_eval_dataset(args):
    dataset = load_dataset(
        args.dataset,
        args.config_name,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000))


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    if args.language:
        language = args.language
    else:
        try:
            language = args.config_name.split("_", 1)[1]
        except IndexError:
            language = "en"
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
            language=language,
            punctuation=args.punctuation,
        )
        audio_chunk_index = inputs.pop("audio_chunk_index", None)
        inputs.pop("length", None)
        if "input_features" in inputs and "input_values" not in inputs:
            inputs["input_values"] = inputs.pop("input_features")
        inputs = inputs.to(device)
        decoder_start_token_id = inputs["decoder_input_ids"][:, 0]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                decoder_start_token_id=decoder_start_token_id,
            )
            pred_text = processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=language,
            )

        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.ml_normalizer(pred, lang=language) for pred in pred_text]
        batch["references"] = [data_utils.ml_normalizer(ref, lang=language) for ref in batch["text"]]
        return batch

    dataset = load_eval_dataset(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    if args.warmup_steps is not None and args.warmup_steps > 0:
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(
            warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True)
        )

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

        dataset = load_eval_dataset(args)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
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

    filtered = [
        (ref, pred, dur, time_s)
        for ref, pred, dur, time_s in zip(
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        (
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
        ) = zip(*filtered)
        all_results = {key: list(values) for key, values in all_results.items()}

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset,
        args.config_name,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer_refs, wer_preds = normalize_compound_pairs(
        all_results["references"], all_results["predictions"]
    )
    wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
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
        "--dataset",
        type=str,
        required=True,
        help="Dataset repository to evaluate, for example `nithinraok/asr-leaderboard-datasets`.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Dataset config name, for example `fleurs_de`.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="ISO 639-1 language code. Defaults to the suffix in `config_name`.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset.",
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
        default=5,
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
