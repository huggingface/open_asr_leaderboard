import argparse
from typing import Optional
import datasets
from datasets import Dataset
import evaluate
import soundfile as sf
import time
import tempfile
import os
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from normalizer import data_utils
import getpass
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    exit(1)


def transcribe_with_retry(
    model_id: str,
    audio_file_path: str,
    sample: Optional[dict] = None,
    max_retries=10,
    use_url=False, # This is not used in the ml script, but we keep it for consistency with the function signature
) -> str:
    retries = 0
    while retries <= max_retries:
        try:
            if model_id.startswith("gemini/"):
                _model = model_id.split("/", 1)[1]
                model = genai.GenerativeModel(_model)

                # In the multilingual script, we always have a local file
                # so we don't need to handle the use_url case for downloading.

                # Upload the file to the Gemini API
                gemini_file = genai.upload_file(path=audio_file_path)

                # Transcribe the audio
                response = model.generate_content(["Generate a transcript of the speech.", gemini_file])

                # Clean up the uploaded file
                genai.delete_file(gemini_file.name)

                return response.text.strip() if response.text else ""
            else:
                raise ValueError(
                    "Invalid model prefix, must start with 'gemini/'"
                )

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            # Re-writing the file on failure is handled by the caller.
            delay = min(2 ** retries, 30)  # Exponential backoff with max 30s
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)
    
    # This should never be reached, but adding for type safety
    return ""


def transcribe_dataset(
    dataset,
    config_name,
    language,
    split,
    model_id,
    max_eval_samples,
    max_workers,
    streaming,
):

    print(f"Loading dataset: {dataset} with config: {config_name}")
    # Force non-streaming to avoid torchcodec issues
    ds = datasets.load_dataset(dataset, config_name, split=split, streaming=False)

    # Apply subsampling first, then convert to pandas
    if max_eval_samples is not None and max_eval_samples > 0:
        print(f"Subsampling dataset to first {max_eval_samples} samples...")
        if hasattr(ds, 'select') and hasattr(ds, '__len__'):
            ds = ds.select(range(min(max_eval_samples, len(ds))))
        else:
            # Fallback: convert to list and slice
            dataset_list = list(ds)
            dataset_list = dataset_list[:max_eval_samples]
            ds = Dataset.from_list(dataset_list)
    
    # Convert to pandas DataFrame to avoid audio decoding during iteration
    print("Converting to pandas DataFrame...")
    try:
        df = ds.to_pandas()
        print(f"Successfully loaded {len(df)} samples")
        print(f"Dataset columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error converting to pandas: {e}")
        # Fallback to list conversion
        dataset_list = []
        for i, sample in enumerate(ds):
            if max_eval_samples and i >= max_eval_samples:
                break
            dataset_list.append(sample)
        df = pd.DataFrame(dataset_list)
        print(f"Successfully loaded {len(df)} samples via fallback")
        print(f"Dataset columns: {df.columns.tolist()}")

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    print(f"Transcribing with model: {model_id}")
    print(f"Processing samples...")
    
    # Process samples sequentially using pandas approach
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        try:
            reference = str(row.get("text", "")).strip() or " "
            
            # Handle audio data similar to main script
            audio_data = row.get('audio')
            
            if audio_data is None:
                print(f"Skipping sample {idx} - no audio data")
                continue
            
            # Handle different audio formats from pandas
            if isinstance(audio_data, dict):
                if 'bytes' in audio_data and audio_data['bytes'] is not None:
                    # Handle bytes format first (most reliable)
                    try:
                        with io.BytesIO(audio_data['bytes']) as audio_file:
                            audio_array, sample_rate = sf.read(audio_file, dtype="float32")
                    except Exception as e:
                        print(f"Error loading audio from bytes: {e}")
                        continue
                elif 'array' in audio_data and audio_data['array'] is not None:
                    audio_array = np.array(audio_data['array'], dtype=np.float32)
                    sample_rate = audio_data.get('sampling_rate', 16000)
                elif 'path' in audio_data and audio_data['path'] is not None:
                    # Load from file path (last resort)
                    try:
                        audio_array, sample_rate = sf.read(audio_data['path'], dtype="float32")
                    except Exception as e:
                        print(f"Error loading audio from path {audio_data['path']}: {e}")
                        continue
                else:
                    print(f"Skipping sample {idx} - unsupported audio format. Available keys: {list(audio_data.keys())}")
                    continue
            else:
                print(f"Skipping sample {idx} - unexpected audio format: {type(audio_data)}")
                continue

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    sf.write(
                        tmpfile.name,
                        audio_array,
                        sample_rate,
                        format="WAV",
                    )
                    tmp_path = tmpfile.name
                    audio_duration = len(audio_array) / sample_rate

                start = time.time()
                try:
                    transcription = transcribe_with_retry(model_id, tmp_path, row)
                    if transcription:
                        results["predictions"].append(transcription)
                        results["references"].append(reference)
                        results["audio_length_s"].append(audio_duration)
                        transcription_time = time.time() - start
                        results["transcription_time_s"].append(transcription_time)
                        print(f"Sample {idx+1}: Transcribed successfully")
                except Exception as e:
                    print(f"Failed to transcribe sample {idx} after retries: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    if language == "en":
        results["predictions"] = [
            data_utils.normalizer(transcription) or " "
            for transcription in results["predictions"]
        ]
        results["references"] = [
            data_utils.normalizer(reference) or " " for reference in results["references"]
        ]
    else:
        results["predictions"] = [
            data_utils.ml_normalizer(transcription) or " "
            for transcription in results["predictions"]
        ]
        results["references"] = [
            data_utils.ml_normalizer(reference) or " " for reference in results["references"]
        ]


    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        model_id,
        dataset,
        config_name,
        split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
    )

    print("Results saved at path:", manifest_path)

    # Only compute WER if we have samples
    if len(results["predictions"]) > 0 and len(results["references"]) > 0:
        wer_metric = evaluate.load("wer")
        wer_result = wer_metric.compute(
            references=results["references"], predictions=results["predictions"]
        )
        wer_value = wer_result if isinstance(wer_result, (int, float)) else wer_result.get('wer', 0.0)
        wer_percent = round(100 * wer_value, 2)
        total_transcription_time = sum(results["transcription_time_s"])
        rtfx = round(
            sum(results["audio_length_s"]) / total_transcription_time, 2
        ) if total_transcription_time > 0 else 0

        print("WER:", wer_percent, "%")
        print("RTFx:", rtfx)
    else:
        print("No samples were successfully processed - cannot compute WER")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier, must start with 'gemini/'",
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
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.set_defaults(streaming=True)
    args = parser.parse_args()

    if args.language is None:
        try:
            args.language = args.config_name.split('_', 1)[1]
        except IndexError:
            raise ValueError("Language could not be inferred from config_name. Please specify it with --language.")

    print(f"Detected language: {args.language}")

    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        gemini_api_key = getpass.getpass("Enter your Gemini API key: ")
    genai.configure(api_key=gemini_api_key)

    transcribe_dataset(
        dataset=args.dataset,
        config_name=args.config_name,
        language=args.language,
        split=args.split,
        model_id=args.model_id,
        max_eval_samples=args.max_eval_samples,
        max_workers=args.max_workers,
        streaming=args.streaming,
    )
