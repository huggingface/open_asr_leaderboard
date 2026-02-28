import argparse
import io
import os
import time
import getpass

import evaluate
import datasets
import numpy as np
import soundfile as sf
from tqdm import tqdm

from normalizer import data_utils

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    exit(1)


def transcribe_with_retry(
    model_id: str,
    audio_file_path: str,
    max_retries: int = 10,
) -> str:
    retries = 0
    while retries <= max_retries:
        try:
            if model_id.startswith("gemini/"):
                _model = model_id.split("/", 1)[1]
                model = genai.GenerativeModel(_model)

                gemini_file = genai.upload_file(path=audio_file_path)
                response = model.generate_content([
                    "Generate a transcript of the speech.",
                    gemini_file,
                ])
                genai.delete_file(gemini_file.name)

                return response.text.strip() if getattr(response, "text", None) else ""
            else:
                raise ValueError("Invalid model prefix, must start with 'gemini/'")

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            delay = min(2 ** retries, 30)  # Exponential backoff with max 30s
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)
    
    # This should never be reached, but adding for type safety
    return ""


def main(args):
    DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")
    CACHE_DIR = os.path.join(DATA_CACHE_DIR, args.dataset, args.split)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load dataset without triggering audio decoding (avoid torchcodec)
    ds = datasets.load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=False,
        token=True,
    )
    # Keep audio as filepaths to avoid decoding here
    try:
        from datasets import Audio
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    # Subsample
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if hasattr(ds, "select") and hasattr(ds, "__len__"):
            ds = ds.select(range(min(args.max_eval_samples, len(ds))))

    references = []
    audio_paths = []
    durations = []

    for sample in tqdm(ds, desc="Preparing samples"):
        sid = str(sample.get("id", "sample")).replace("/", "_").removesuffix(".wav")
        audio_info = sample.get("audio")
        if not isinstance(audio_info, dict):
            print("Skipping sample without audio info")
            continue
        try:
            if audio_info.get("bytes") is not None:
                with io.BytesIO(audio_info["bytes"]) as bio:
                    audio_array, sr = sf.read(bio, dtype="float32")
            elif audio_info.get("path"):
                audio_array, sr = sf.read(audio_info["path"], dtype="float32")
            elif audio_info.get("array") is not None:
                audio_array = np.float32(audio_info["array"]) if not isinstance(audio_info["array"], np.ndarray) else audio_info["array"].astype(np.float32)
                sr = audio_info.get("sampling_rate", 16000)
            else:
                print("Skipping sample: unsupported audio format")
                continue
        except Exception as e:
            print(f"Failed to read audio: {e}")
            continue

        out_path = os.path.join(CACHE_DIR, f"{sid}.wav")
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio_array, sr)

        audio_paths.append(out_path)
        durations.append(len(audio_array) / sr)

        # Normalize reference text
        try:
            ref_text = data_utils.get_text(sample)
        except Exception:
            ref_text = sample.get("text", " ")
        references.append(data_utils.normalizer(ref_text) or " ")

        if args.max_eval_samples is not None and len(audio_paths) >= args.max_eval_samples:
            break

    # Transcribe
    predictions = []
    transcription_times = []
    print(f"Transcribing with model: {args.model_id}")
    for audio_path in tqdm(audio_paths, desc="Transcribing"):
        start = time.time()
        try:
            pred_text = transcribe_with_retry(args.model_id, audio_path)
        except Exception as e:
            print(f"Failed to transcribe {audio_path}: {e}")
            pred_text = " "
        elapsed = time.time() - start
        transcription_times.append(elapsed)
        predictions.append(data_utils.normalizer(pred_text) or " ")
        time.sleep(0.1)

    if len(predictions) == 0:
        print("No samples were successfully processed.")
        return

    manifest_path = data_utils.write_manifest(
        references,
        predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=durations,
        transcription_time=transcription_times,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    rtfx = round(sum(durations) / max(1e-9, sum(transcription_times)), 2)
    print("WER:", wer, "%")
    print("RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini ASR Evaluation Script")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier, must start with 'gemini/'",
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
        help="Dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples per streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Download the entire dataset instead of streaming.",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = getpass.getpass("Enter your Gemini API key: ")
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set and no key provided interactively.")
    genai.configure(api_key=api_key)

    main(args)
