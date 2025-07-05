import argparse
from typing import Optional
import datasets
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
import itertools
from tqdm import tqdm
from dotenv import load_dotenv
import aiohttp
import uuid
import asyncio
from normalizer import data_utils
import concurrent.futures
import aiofiles

load_dotenv()


def fetch_audio_urls(dataset_path, dataset, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={dataset}&split={split}"
    size_response = requests.get(size_url).json()
    total_rows = size_response["size"]["config"]["num_rows"]
    audio_urls = []
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_rows - offset),
        }

        retries = 0
        while retries <= max_retries:
            try:
                headers = {}
                if os.environ.get("HF_TOKEN") is not None:
                    headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
                else:
                    print("HF_TOKEN not set, might experience rate-limiting.")
                response = requests.get(API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                yield from data["rows"]
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                retries += 1
                print(
                    f"Error fetching data: {e}, retrying ({retries}/{max_retries})..."
                )
                time.sleep(10)
                if retries >= max_retries:
                    raise Exception("Max retries exceeded while fetching data.")


async def transcribe_shunyaai(
    model_name: str,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    api_base_url: str = "https://pingala.shunyalabs.ai",
) -> str:
    """
    Transcribe audio using your FastAPI transcription service

    Args:
        model_name: Model name (not used since it's configured in the API)
        file_path: Path to local audio file
        url: URL to download audio file from
        api_base_url: Base URL of your transcription API

    Returns:
        Transcribed text
    """
    temp_file = None

    if not file_path and not url:
        raise ValueError("Either file_path or url must be provided.")

    try:
        # Download file if URL is provided
        if url:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    temp_file = os.path.join(
                        tempfile.gettempdir(), f"temp_{uuid.uuid4()}.wav"
                    )

                    async with aiofiles.open(temp_file, "wb") as f:
                        await f.write(await resp.read())

                    file_path = temp_file

        # Send file to your FastAPI transcription service
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as audio_file:
                # Create form data
                data = aiohttp.FormData()
                data.add_field("file", audio_file, filename=os.path.basename(file_path))

                # Optional: Add query parameters for chunking settings
                params = {
                    "chunk_length_s": 30,  # Adjust as needed
                    "overlap_s": 2,  # Adjust as needed
                    "max_file_size_mb": 500,  # Adjust as needed
                }

                # Include API key header if available
                headers = {}
                api_key = os.getenv("SHUNYA_API_KEY")
                if api_key:
                    headers["x-api-key"] = api_key

                # Make request to your API
                async with session.post(
                    f"{api_base_url}/transcribe",
                    data=data,
                    params=params,
                    headers=headers,  # Added headers containing x-api-key when provided
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()

                    # Extract transcribed text
                    return result.get("text", "")

    except aiohttp.ClientError as e:
        raise RuntimeError(f"API request failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
):
    retries = 0
    while retries <= max_retries:
        try:
            if model_name.startswith("shunyalabs/"):
                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    transcription = asyncio.run(
                        transcribe_shunyaai(model_name, url=audio_url)
                    )
                else:
                    transcription = asyncio.run(
                        transcribe_shunyaai(model_name, file_path=audio_file_path)
                    )
                return transcription

            else:
                raise ValueError("Invalid model prefix, must start with 'shunyalabs/'")

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            if not use_url:
                sf.write(
                    audio_file_path,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
            delay = 1
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)


def transcribe_dataset(
    dataset_path,
    dataset,
    split,
    model_name,
    use_url=False,
    max_samples=None,
    max_workers=4,
):
    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, dataset, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = audio_rows
    else:
        ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)
        ds = data_utils.prepare_data(ds)
        if max_samples:
            ds = ds.take(max_samples)

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    print(f"Transcribing with model: {model_name}")

    def process_sample(sample):
        if use_url:
            reference = sample["row"]["text"].strip() or " "
            audio_duration = sample["row"]["audio_length_s"]
            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, None, sample, use_url=True
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                return None

        else:
            reference = sample.get("norm_text", "").strip() or " "
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
                    model_name, tmp_path, sample, use_url=False
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

    results["predictions"] = [
        data_utils.normalizer(transcription) or " "
        for transcription in results["predictions"]
    ]
    results["references"] = [
        data_utils.normalizer(reference) or " " for reference in results["references"]
    ]

    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        model_name.replace("/", "-"),
        dataset_path,
        dataset,
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
    parser = argparse.ArgumentParser(
        description="Unified Transcription Script with Concurrency"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Prefix model name with 'shunyaai/'",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_workers", type=int, default=300, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--use_url",
        action="store_true",
        help="Use URL-based audio fetching instead of datasets",
    )

    args = parser.parse_args()

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        use_url=args.use_url,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
    )
