import argparse
from typing import Optional
import datasets
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
from tqdm import tqdm
from normalizer import data_utils
import concurrent.futures
import getpass
import google.generativeai as genai


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False, # This is not used in the ml script, but we keep it for consistency with the function signature
):
    retries = 0
    while retries <= max_retries:
        try:
            if model_name.startswith("gemini/"):
                model_id = model_name.split("/", 1)[1]
                model = genai.GenerativeModel(model_id)

                # In the multilingual script, we always have a local file
                # so we don't need to handle the use_url case for downloading.

                # Upload the file to the Gemini API
                gemini_file = genai.upload_file(path=audio_file_path)

                # Transcribe the audio
                response = model.generate_content(["Generate a transcript of the speech.", gemini_file])

                # Clean up the uploaded file
                genai.delete_file(gemini_file.name)

                return response.text.strip()
            else:
                raise ValueError(
                    "Invalid model prefix, must start with 'gemini/'"
                )

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            # Re-writing the file on failure is handled by the caller.
            delay = 1
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)


def transcribe_dataset(
    dataset,
    config_name,
    language,
    split,
    model_name,
    max_samples,
    max_workers,
    streaming,
):

    print(f"Loading dataset: {dataset} with config: {config_name}")
    ds = datasets.load_dataset(dataset, config_name, split=split, streaming=streaming)

    if max_samples is not None and max_samples > 0:
        print(f"Subsampling dataset to first {max_samples} samples!")
        if streaming:
            ds = ds.take(max_samples)
        else:
            ds = ds.select(range(min(max_samples, len(ds))))


    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    print(f"Transcribing with model: {model_name}")

    def process_sample(sample):
        reference = sample.get("text", "").strip() or " "

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(
                tmpfile.name,
                sample["audio"]["array"],
                sample["audio"]["sampling_rate"],
                format="WAV",
            )
            tmp_path = tmpfile.name
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )

        start = time.time()
        try:
            transcription = transcribe_with_retry(
                model_name, tmp_path, sample
            )
        except Exception as e:
            print(f"Failed to transcribe after retries: {e}")
            os.unlink(tmp_path)
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            else:
                print(f"File {tmp_path} does not exist")

        transcription_time = time.time() - start
        return reference, transcription, audio_duration, transcription_time

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(process_sample, sample): sample for sample in ds
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sample),
            total=len(future_to_sample),
            desc="Transcribing",
        ):
            result = future.result()
            if result:
                reference, transcription, audio_duration, transcription_time = result
                results["predictions"].append(transcription)
                results["references"].append(reference)
                results["audio_length_s"].append(audio_duration)
                results["transcription_time_s"].append(transcription_time)

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
        model_name.replace("/", "-"),
        dataset,
        config_name,
        split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
    )

    print("Results saved at path:", manifest_path)

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(
        references=results["references"], predictions=results["predictions"]
    )
    wer_percent = round(100 * wer, 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, required=True, help="Model identifier. Should be loadable with Gemini.",
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
        "--max_samples",
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
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    if args.language is None:
        try:
            args.language = args.config_name.split('_', 1)[1]
        except IndexError:
            raise ValueError("Language could not be inferred from config_name. Please specify it with --language.")

    print(f"Detected language: {args.language}")

    gemini_api_key = getpass.getpass("Enter your Gemini API key: ")
    genai.configure(api_key=gemini_api_key)

    transcribe_dataset(
        dataset=args.dataset,
        config_name=args.config_name,
        language=args.language,
        split=args.split,
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        streaming=args.streaming,
    )
