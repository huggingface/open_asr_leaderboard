import argparse
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
from datasets import load_dataset, Audio

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


def main(args):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_input_name = processor.model_input_names[0]

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}

    # For multilingual Whisper models, set task to transcribe but let language auto-detect
    if getattr(model.generation_config, "is_multilingual", False):
        gen_kwargs["task"] = "transcribe"
    else:
        print(f"Warning: Model {args.model_id} is not multilingual.")

    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        model.generation_config.cache_implementation = "static"

    # Load dataset
    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    def benchmark(batch, min_new_tokens=None):
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        # Standard Whisper processing: pad audios to 30-seconds and convert to log-mel
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", device=args.device)
        inputs = inputs.to(args.device)
        inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)

        # Model Inference
        pred_ids = model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)

        # Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # Convert token ids to text transcription
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # END TIMING
        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Normalize with multilingual normalizer
        batch["predictions"] = [data_utils.ml_normalizer(pred) for pred in pred_text]
        batch["references"] = [data_utils.ml_normalizer(ref) for ref in batch["text"]]

        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(
            benchmark, batch_size=args.batch_size, batched=True,
            fn_kwargs={"min_new_tokens": args.max_new_tokens}
        ))
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Reload dataset for actual evaluation (reset streaming pointer)
    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
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
        args.dataset,
        CONFIG_NAME,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
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
        help="Model identifier. Should be loadable with Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. E.g. 'nithinraok/asr-leaderboard-datasets'",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name for the dataset. E.g. 'fleurs_de' for German FLEURS.",
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
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of samples to go through each batch.",
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
        default=None,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to JIT compile the forward pass of the model.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        help="Mode for torch compiling model forward pass.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
