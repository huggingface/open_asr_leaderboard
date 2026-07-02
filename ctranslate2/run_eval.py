"""Run evaluation for ctranslate2 whisper models."""""
import argparse
import os
import time

import evaluate
from faster_whisper import WhisperModel
from tqdm import tqdm

from normalizer import data_utils

wer_metric = evaluate.load("wer")


def main(args) -> None:
    """Main function to run evaluation on a dataset."""
    asr_model = WhisperModel(
        model_size_or_path=args.model_id,
        compute_type="float16",
        device="cuda",
        device_index=args.device
    )

    def benchmark(batch):
        start_time = time.time()
        segments, _ = asr_model.transcribe(batch["audio"]["array"], language="en")
        outputs = [segment._asdict() for segment in segments]
        batch["transcription_time_s"] = time.time() - start_time
        batch["predictions"] = "".join([segment["text"] for segment in outputs]).strip()  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        if args.streaming:
            warmup_dataset = dataset.take(args.warmup_steps)
        else:
            warmup_dataset = dataset.select(range(min(args.warmup_steps, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, remove_columns=["audio"]))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(benchmark, remove_columns=["audio"])

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

    norm_refs = [data_utils.normalizer(r) for r in all_results["references"]]
    norm_preds = [data_utils.normalizer(p) for p in all_results["predictions"]]
    wer = wer_metric.compute(
        references=norm_refs, predictions=norm_preds
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
        help="Model identifier. Should be loadable with faster-whisper",
    )
    parser.add_argument(
        '--dataset_path', type=str, default='hf-audio/open-asr-leaderboard', help='Dataset path. By default, it is `hf-audio/open-asr-leaderboard`'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
            "can be found at `https://huggingface.co/datasets/hf-audio/open-asr-leaderboard`"
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
