import argparse
import os

import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import evaluate
from normalizer import data_utils
from datasets import load_dataset, Audio
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")

# Mapping from ISO 639-1 language codes to omniASR language codes (BCP 47 / Flores-200 style)
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "el": "ell_Grek",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mt": "mlt_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sv": "swe_Latn",
    "uk": "ukr_Cyrl",
}


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

    # Map to omniASR language code
    omniasr_lang = LANG_CODE_MAP.get(LANGUAGE, "eng_Latn")
    print(f"Language: {LANGUAGE} -> omniASR lang code: {omniasr_lang}")

    # Always use the multilingual normalizer
    text_normalizer = data_utils.ml_normalizer

    # Map model_id to model_card format expected by omnilingual_asr
    # e.g., "facebook/omniASR-LLM-7B" -> "omniASR_LLM_7B"
    model_card = args.model_id.split("/")[-1].replace("-", "_")

    # Initialize the ASR pipeline
    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    pipeline = ASRInferencePipeline(
        model_card=model_card,
        device=device
    )

    MAX_AUDIO_SEC = 40  # Pipeline max audio length

    def get_text(sample):
        if "text" in sample:
            return sample["text"]
        elif "sentence" in sample:
            return sample["sentence"]
        elif "normalized_text" in sample:
            return sample["normalized_text"]
        elif "transcript" in sample:
            return sample["transcript"]
        elif "transcription" in sample:
            return sample["transcription"]
        else:
            raise ValueError(
                f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. "
                f"Got sample keys: {list(sample.keys())}"
            )

    def benchmark(batch):
        minibatch_size = len(batch["audio"])

        # Convert to pipeline input format
        audio_data = []
        for audio in batch["audio"]:
            waveform = audio["array"]
            sample_rate = audio["sampling_rate"]
            max_samples = int(MAX_AUDIO_SEC * sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            audio_data.append({"waveform": waveform, "sample_rate": sample_rate})

        # Compute audio lengths
        batch["audio_length_s"] = [
            len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]
        ]

        # START TIMING
        start_time = time.time()

        # Inference with the appropriate language code
        lang = [omniasr_lang] * minibatch_size
        transcriptions = pipeline.transcribe(
            audio_data,
            # lang=lang,
            batch_size=minibatch_size
        )

        # END TIMING
        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Normalize with appropriate normalizer
        batch["predictions"] = [text_normalizer(pred) for pred in transcriptions]

        # Get references and normalize
        references = []
        for i in range(minibatch_size):
            sample = {k: batch[k][i] for k in batch if k != "audio"}
            ref = get_text(sample)
            references.append(text_normalizer(ref))
        batch["references"] = references

        return batch

    # Load dataset
    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Filter out empty references
    dataset = dataset.filter(
        lambda x: data_utils.is_target_text_in_range(get_text(x))
    )

    # Warmup
    if args.warmup_steps is not None:
        warmup_dataset = load_dataset(
            args.dataset,
            CONFIG_NAME,
            split=SPLIT_NAME,
            streaming=args.streaming,
            token=True,
        )
        warmup_dataset = warmup_dataset.cast_column("audio", Audio(sampling_rate=16000))
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Run evaluation
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

    # Write manifest results
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset,
        CONFIG_NAME,
        SPLIT_NAME,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print(f"Dataset: {args.dataset}")
    print(f"Language: {LANGUAGE}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Model: {args.model_id}")
    print(f"WER: {wer}%")
    print(f"RTFx: {rtfx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier on Hugging Face (e.g., 'facebook/omniASR-LLM-7B')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nithinraok/asr-leaderboard-datasets",
        help="Dataset path. Default is 'nithinraok/asr-leaderboard-datasets'",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name in format <dataset>_<lang> (e.g., fleurs_en, mcv_de, mls_es)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, de, es). If not provided, will be extracted from config_name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. Default is 'test'.",
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
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
