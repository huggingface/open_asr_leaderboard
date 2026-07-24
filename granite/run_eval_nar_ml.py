import argparse
import os

# Disable strict type validation in huggingface_hub (model config has int where float expected)
os.environ["HF_HUB_DISABLE_STRICT_FIELD_VALIDATION"] = "1"

import torch
import evaluate
from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs
import time
from tqdm import tqdm
from datasets import load_dataset, Audio

from transformers import AutoModel, AutoProcessor

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


def main(args):
    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split

    # Extract language from config_name if not provided
    if args.language:
        LANGUAGE = args.language
    else:
        try:
            LANGUAGE = CONFIG_NAME.split("_", 1)[1]
        except IndexError:
            LANGUAGE = "en"

    # Load model (NLENARDecoder requires flash_attention_2, no fallback possible)
    device = f"cuda:{args.device}" if args.device != -1 else "cpu"
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True,
                                      revision=args.revision,
                                      attn_implementation="flash_attention_2",
                                      device_map=device, dtype=torch.bfloat16).eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True,
                                              revision=args.revision)

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

    def benchmark(batch):
        # Load audio inputs
        audios = [torch.tensor(audio["array"], device=device).squeeze(0) for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]]

        # START TIMING
        start_time = time.time()
        inputs = processor(audios, device=device)
        # Model Inference
        with torch.inference_mode():
            output = model.transcribe(**inputs)
            output_text = processor.batch_decode(output.preds)
        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = output_text  # raw; normalization applied at scoring time
        batch["references"] = batch["text"]  # raw; normalization applied at scoring time
        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Reload dataset for actual evaluation
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

    # Filter empty references (consistent with English pipeline)
    filtered = [
        (ref, pred, dur, time_s)
        for ref, pred, dur, time_s in zip(
            all_results["references"], all_results["predictions"],
            all_results["audio_length_s"], all_results["transcription_time_s"],
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        all_results["references"], all_results["predictions"], all_results["audio_length_s"], all_results["transcription_time_s"] = zip(*filtered)
        all_results = {k: list(v) for k, v in all_results.items()}

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

    norm_refs = [data_utils.ml_normalizer(r, lang=LANGUAGE) for r in all_results["references"]]
    norm_preds = [data_utils.ml_normalizer(p, lang=LANGUAGE) for p in all_results["predictions"]]
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="ibm-granite/granite-speech-4.1-2b-nar",
        help="Model identifier. Should be loadable with Transformers",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (commit SHA or tag) for trust_remote_code safety.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. E.g. 'hf-audio/open-asr-leaderboard-multilingual-datasets'",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name for the dataset. E.g. 'fleurs_de' for German FLEURS.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'de'). If not provided, extracted from config_name.",
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
        default=128,
        help="Number of samples to go through each batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
