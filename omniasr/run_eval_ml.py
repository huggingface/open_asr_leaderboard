import argparse
import json
import os
import re

import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import evaluate
from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs
from datasets import load_dataset, Audio
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")

# omnilingual_asr expects NLLB-style language codes (e.g. "deu_Latn"), not the
# 2-letter codes used by the datasets/normalizer.
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "hi": "hin_Deva",
}

def main(args):
    CONFIG_NAME = args.config_name  # None for single-config dataset repos (e.g. VoiceArena/Monsoon_hi_test)
    SPLIT_NAME = args.split

    # Determine language for normalization: use --language if provided, otherwise
    # extract from config_name (e.g. "fleurs_de") or, for single-config repos,
    # from the dataset name (e.g. "Monsoon_hi_test").
    if args.language:
        LANGUAGE = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        LANGUAGE = lang_match.group(1) if lang_match else "en"

    # Always use the multilingual normalizer with number normalization
    text_normalizer = lambda s: data_utils.ml_normalizer(s, lang=LANGUAGE)

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
        if "lattice" in sample:
            # Lattice reference (e.g. VoiceArena/Monsoon_hi_test): JSON-encode the
            # lattice so it survives the manifest; scoring decodes it and uses
            # voi_oiwer (see normalizer/eval_utils.py).
            return json.dumps(sample["lattice"], ensure_ascii=False)
        elif "text" in sample:
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
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        # START TIMING
        start_time = time.time()

        # Force the target language (NLLB code, e.g. "deu_Latn"), consistent
        # with the API models which always pass the language to the provider.
        lang = [NLLB_LANGUAGE_CODES.get(LANGUAGE, "eng_Latn")] * minibatch_size
        transcriptions = pipeline.transcribe(
            audio_data,
            lang=lang,
            batch_size=minibatch_size
        )

        # END TIMING
        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Normalize with appropriate normalizer
        batch["predictions"] = transcriptions  # raw; normalization applied at scoring time

        # Get raw references
        batch["references"] = [
            get_text({k: batch[k][i] for k in batch if k != "audio"})
            for i in range(minibatch_size)
        ]  # raw; normalization applied at scoring time

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
        "audio_filepath": [],
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
        CONFIG_NAME or "",
        SPLIT_NAME,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    from normalizer.eval_utils import OIWER_LANGUAGES, score_oiwer
    if LANGUAGE in OIWER_LANGUAGES:
        # Lattice-based, orthography-aware scoring (voi_oiwer applies its own
        # normalization internally).
        manifest = [
            {"text": ref, "pred_text": pred}
            for ref, pred in zip(all_results["references"], all_results["predictions"])
        ]
        wer, _ins, _del, _sub = score_oiwer(manifest, OIWER_LANGUAGES[LANGUAGE])
        wer = round(100 * wer, 2)
    else:
        norm_refs = [text_normalizer(r) for r in all_results["references"]]
        norm_preds = [text_normalizer(p) for p in all_results["predictions"]]
        wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
        wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
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
        default="hf-audio/open-asr-leaderboard-multilingual-datasets",
        help="Dataset path. Default is 'hf-audio/open-asr-leaderboard-multilingual-datasets'",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name in format <dataset>_<lang> (e.g., fleurs_en, mcv_de, mls_es). "
             "Omit for single-config dataset repos (e.g. 'VoiceArena/Monsoon_hi_test').",
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
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
