# This script is used to evaluate NeMo ASR models on the Multi-Lingual datasets

import argparse
import io
import os
import torch
import evaluate
import soundfile
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from normalizer import data_utils
from nemo.collections.asr.models import ASRModel
import time


wer_metric = evaluate.load("wer")


def main(args):
    DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")
    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split
    
    # Extract language from config_name if not provided
    if args.language:
        LANGUAGE = args.language
    else:
        # Extract language from config_name (e.g., "fleurs_en" -> "en")
        try:
            LANGUAGE = CONFIG_NAME.split('_', 1)[1]
        except IndexError:
            LANGUAGE = "en"  # Default fallback
    
    print(f"Detected language: {LANGUAGE}")

    CACHE_DIR = os.path.join(DATA_CACHE_DIR, CONFIG_NAME, SPLIT_NAME)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        compute_dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32

    # Load ASR model
    if args.model_id.endswith(".nemo"):
        asr_model = ASRModel.restore_from(args.model_id, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(args.model_id, map_location=device)
    
    asr_model.to(compute_dtype)
    asr_model.eval()

    # Load dataset using the HuggingFace dataset repository
    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")

    dataset = load_dataset(args.dataset, CONFIG_NAME, split=SPLIT_NAME, streaming=args.streaming)
    
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    # Configure decoding strategy
    if asr_model.cfg.decoding.strategy != "beam":
        asr_model.cfg.decoding.strategy = "greedy_batch"
        asr_model.change_decoding_strategy(asr_model.cfg.decoding)

    def download_audio_files(batch):
        """Process audio files and prepare them for evaluation."""
        audio_paths = []
        durations = []

        for i, (file_name, sample, duration, text) in enumerate(zip(
            batch["file_name"], batch["audio"], batch["duration"], batch["text"]
        )):
            # Create unique filename using index to avoid conflicts
            unique_id = f"{CONFIG_NAME}_{i}_{os.path.basename(file_name).replace('.wav', '')}"
            audio_path = os.path.join(CACHE_DIR, f"{unique_id}.wav")

            if "array" in sample:
                audio_array = np.float32(sample["array"])
                sample_rate = sample.get("sampling_rate", 16000)
            elif "bytes" in sample:
                with io.BytesIO(sample["bytes"]) as audio_file:
                    audio_array, sample_rate = soundfile.read(audio_file, dtype="float32")
            else:
                raise ValueError("Sample must have either 'array' or 'bytes' key")

            if not os.path.exists(audio_path):
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                soundfile.write(audio_path, audio_array, sample_rate)

            audio_paths.append(audio_path)
            # Use duration from dataset if available, otherwise calculate
            if duration is not None:
                durations.append(duration)
            else:
                durations.append(len(audio_array) / sample_rate)

        batch["references"] = [text for text in batch["text"]]
        batch["audio_filepaths"] = audio_paths
        batch["durations"] = durations

        return batch

    # Process the dataset
    print("Processing audio files...")
    dataset = dataset.map(
        download_audio_files, 
        batch_size=args.batch_size, 
        batched=True, 
        remove_columns=["audio"]
    )

    # Collect all data
    all_data = {
        "audio_filepaths": [],
        "durations": [],
        "references": [],
    }

    print("Collecting data...")
    for data in tqdm(dataset, desc="Collecting samples"):
        all_data["audio_filepaths"].append(data["audio_filepaths"])
        all_data["durations"].append(data["durations"])
        all_data["references"].append(data["references"])

    # Sort by duration for efficient batch processing
    print("Sorting by duration...")
    sorted_indices = sorted(range(len(all_data["durations"])), key=lambda k: all_data["durations"][k], reverse=True)
    all_data["audio_filepaths"] = [all_data["audio_filepaths"][i] for i in sorted_indices]
    all_data["references"] = [all_data["references"][i] for i in sorted_indices]
    all_data["durations"] = [all_data["durations"][i] for i in sorted_indices]

    # Run evaluation with warmup
    total_time = 0
    for warmup_round in range(2):  # warmup once and calculate rtf
        if warmup_round == 0:
            audio_files = all_data["audio_filepaths"][:args.batch_size * 4]  # warmup with 4 batches
            print("Running warmup...")
        else:
            audio_files = all_data["audio_filepaths"]
            print("Running full evaluation...")
            
        start_time = time.time()
        with torch.inference_mode(), torch.no_grad():
            # for canary-1b and canary-1b-flash, we need to set pnc='no' for English and for other languages, we need to set pnc='pnc' but for canary-1b-v2 pnc='yes' for all languages
            if 'canary' in args.model_id and 'v2' not in args.model_id:
                pnc = 'nopnc' if LANGUAGE == "en" else 'pnc'
            else:
                pnc = 'pnc'

            if 'canary' in args.model_id:
                transcriptions = asr_model.transcribe(audio_files, batch_size=args.batch_size, verbose=False, pnc=pnc, num_workers=1, source_lang=LANGUAGE, target_lang=LANGUAGE)
            else:
                transcriptions = asr_model.transcribe(audio_files, batch_size=args.batch_size, verbose=False, num_workers=1)
        end_time = time.time()
        
        if warmup_round == 1:
            total_time = end_time - start_time

    # Process transcriptions
    if isinstance(transcriptions, tuple) and len(transcriptions) == 2:
        transcriptions = transcriptions[0]
    
    references = all_data["references"] 
    if LANGUAGE == "en": # English is handled by the English normalizer
        references = [data_utils.normalizer(ref) for ref in references]
        predictions = [data_utils.normalizer(pred.text) for pred in transcriptions]
    else:
        references = [data_utils.ml_normalizer(ref) for ref in references]
        predictions = [data_utils.ml_normalizer(pred.text) for pred in transcriptions]

    avg_time = total_time / len(all_data["audio_filepaths"])

    # Write results using eval_utils.write_manifest
    manifest_path = data_utils.write_manifest(
        references,
        predictions,
        args.model_id,
        args.dataset,  # dataset_path for filename
        CONFIG_NAME,  # dataset_name
        SPLIT_NAME,
        audio_length=all_data["durations"],
        transcription_time=[avg_time] * len(all_data["audio_filepaths"]),
    )

    print("Results saved at path:", os.path.abspath(manifest_path))

    # Calculate metrics
    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    audio_length = sum(all_data["durations"])
    rtfx = audio_length / total_time
    rtfx = round(rtfx, 2)

    print(f"Dataset: {args.dataset}")
    print(f"Language: {LANGUAGE}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Model: {args.model_id}")
    print(f"RTFX: {rtfx}")
    print(f"WER: {wer}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with NVIDIA NeMo.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nithinraok/asr-leaderboard-datasets",
        help="Dataset name. Default is 'nithinraok/asr-leaderboard-datasets'"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name in format <dataset>_<lang> (e.g., fleurs_en, mcv_de, mls_es)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, de, es). If not provided, will be extracted from config_name."
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
        "--batch_size", type=int, default=32, help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )

    parser.add_argument(
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args) 