# This script is used to evaluate NeMo ASR models on the Multi-Lingual datasets

import argparse
import os
# Force soundfile audio decoding before datasets is imported/used,
# to avoid the torchcodec AudioDecoder object being returned.
os.environ.setdefault("HF_AUDIO_DECODER_BACKEND", "soundfile")
import torch
import evaluate
import soundfile
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio
from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf
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
    print(f"Model size: {sum(p.numel() for p in asr_model.parameters()) / 1e9:.2f}B parameters")

    # Load dataset using the HuggingFace dataset repository
    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")

    dataset = load_dataset(args.dataset, CONFIG_NAME, split=SPLIT_NAME, streaming=args.streaming)

    # Re-sample and cast audio to a consistent dict format ({"array", "sampling_rate"}),
    # matching run_eval.py's data_utils.prepare_data(). Without this, some `datasets`
    # versions/backends decode audio into a non-subscriptable AudioDecoder object.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    # Configure decoding strategy
    if asr_model.cfg.decoding.strategy != "beam":
        asr_model.cfg.decoding.strategy = "greedy_batch"
        if hasattr(asr_model.cfg.decoding, "greedy"):
            OmegaConf.update(asr_model.cfg.decoding, "greedy.use_cuda_graph_decoder", False, force_add=True)
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

            audio_array = np.float32(sample["array"])
            sample_rate = sample["sampling_rate"]

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
    
    references = all_data["references"]  # raw
    predictions = [pred.text for pred in transcriptions]  # raw; normalization applied at scoring time

    # Filter empty references (consistent with English pipeline)
    filtered = [
        (ref, pred, dur)
        for ref, pred, dur in zip(references, predictions, all_data["durations"])
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        references, predictions, all_data["durations"] = zip(*filtered)
        references, predictions = list(references), list(predictions)
        all_data["durations"] = list(all_data["durations"])

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
    if LANGUAGE == "en":
        norm_refs = [data_utils.normalizer(r) for r in references]
        norm_preds = [data_utils.normalizer(p) for p in predictions]
    else:
        norm_refs = [data_utils.ml_normalizer(r, lang=LANGUAGE) for r in references]
        norm_preds = [data_utils.ml_normalizer(p, lang=LANGUAGE) for p in predictions]
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
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
        default="hf-audio/open-asr-leaderboard-multilingual-datasets",
        help="Dataset name. Default is 'hf-audio/open-asr-leaderboard-multilingual-datasets'"
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
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    args = parser.parse_args()

    main(args) 