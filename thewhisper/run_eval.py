import argparse
import os
import torch
from transformers import WhisperProcessor, AutoTokenizer
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm

from elastic_models.transformers import WhisperForConditionalGeneration

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


def main(args):
    dtype = torch.float16
    chunk_length = 20

    # Load model using elastic_models for optimized inference
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        chunk_length=chunk_length,
        torch_dtype=dtype,
        mode=args.mode
    ).to(args.device)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.cache_implementation = "flexi-static"

    # Load processor with matching chunk_length
    processor = WhisperProcessor.from_pretrained(
        args.model_id,
        chunk_length=chunk_length,
        tokenizer=AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    )
    model_input_name = processor.model_input_names[0]

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "do_sample": False,
        "task": "transcribe",
        "language": "en",
        "cache_implementation": "flexi-static",
    }

    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # 1. Pre-Processing
        # Pad audios to max batch size if needed for consistent batching
        padding_size = None
        if minibatch_size != args.batch_size:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        # Process audio with Whisper processor
        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            device=args.device
        )
        inputs = inputs.to(args.device)
        inputs[model_input_name] = inputs[model_input_name].to(dtype)

        # 2. Model Inference
        with torch.no_grad():
            pred_ids = model.generate(
                **inputs,
                **gen_kwargs,
                min_new_tokens=min_new_tokens
            )

        # 3. Post-processing
        # Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # Convert token ids to text transcription
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer from open_asr_leaderboard
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    # Warmup steps
    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(
            benchmark,
            batch_size=args.batch_size,
            batched=True,
            fn_kwargs={"min_new_tokens": args.max_new_tokens}
        ))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Load and prepare data
    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    # Run evaluation
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

    # Write manifest results (WER and RTFX)
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
        references=all_results["references"],
        predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with elastic_models",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="XL",
        choices=["S", "XL"],
        help="Inference mode: 'S' for int8 quantized, 'XL' for fp16 with compiler optimization.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    # Convert device index to proper format
    if args.device >= 0:
        args.device = f"cuda:{args.device}"
    else:
        args.device = "cpu"

    main(args)
