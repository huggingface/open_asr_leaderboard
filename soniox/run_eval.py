import argparse
import asyncio
import json
from typing import Optional
import datasets
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
import websockets
import itertools
from tqdm import tqdm
from dotenv import load_dotenv
from normalizer import data_utils
import concurrent.futures

load_dotenv()

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")

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


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
    mode="async",
    language="en"
):
    if mode == "async":
        return transcribe_async(model_name, audio_file_path, sample, max_retries, use_url, language)
    elif mode == "realtime":
        # Since the transcribe_dataset is called with ThreadPoolExecutor,
        # we need to create a new event loop for each thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(transcribe_realtime(model_name, audio_file_path, sample, max_retries, use_url, language))
        finally:
            loop.close()
    else:
        raise ValueError(f"Invalid mode: {mode}")


def transcribe_async(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
    language="en"
):
    retries = 0
    while retries <= max_retries:
        try:
            api_base = "https://api.soniox.com"
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {SONIOX_API_KEY}"

            if use_url:
                audio_url = sample["row"]["audio"][0]["src"]
                res = session.post(
                    f"{api_base}/v1/transcriptions",
                    json={
                        "audio_url": audio_url,
                        "model": model_name,
                        "language_hints": [language],
                    },
                )
            else:
                res = session.post(
                    f"{api_base}/v1/files",
                    files={"file": open(audio_file_path, "rb")},
                )
                file_id = res.json()["id"]
                res = session.post(
                    f"{api_base}/v1/transcriptions",
                    json={
                        "file_id": file_id,
                        "model": model_name,
                        "language_hints": [language],
                    },
                )

            res.raise_for_status()
            transcription_id = res.json()["id"]

            while True:
                res = session.get(f"{api_base}/v1/transcriptions/{transcription_id}")
                res.raise_for_status()
                data = res.json()
                if data["status"] == "completed":
                    break
                elif data["status"] == "error":
                    raise Exception(f"Transcription failed: {data.get('error_message', 'Unknown error')}")
                time.sleep(1)

            res = session.get(f"{api_base}/v1/transcriptions/{transcription_id}/transcript")
            res.raise_for_status()
            transcript = res.json()["text"]

            # Cleanup
            session.delete(f"{api_base}/v1/transcriptions/{transcription_id}")
            if not use_url:
                session.delete(f"{api_base}/v1/files/{file_id}")

            return transcript

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e
            delay = 1
            print(f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
            time.sleep(delay)


async def transcribe_realtime(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
    language="en"
):
    retries = 0
    while retries <= max_retries:
        try:
            websocket_url = "wss://stt-rt.soniox.com/transcribe-websocket"
            async with websockets.connect(websocket_url) as websocket:
                init_msg = {
                    "api_key": SONIOX_API_KEY,
                    "model": model_name,
                    "sample_rate": 16000,
                    "language_hints": [language],
                }
                await websocket.send(json.dumps(init_msg))

                if use_url:
                    response = requests.get(sample["row"]["audio"][0]["src"])
                    audio_data = response.content
                else:
                    with open(audio_file_path, "rb") as f:
                        audio_data = f.read()

                for i in range(0, len(audio_data), 1024):
                    await websocket.send(audio_data[i:i+1024])
                await websocket.send(b"")

                transcript = ""
                async for message in websocket:
                    data = json.loads(message)
                    if "tokens" in data:
                        for token in data["tokens"]:
                            if token.get("is_final"):
                                transcript += token.get("text", "")
                return transcript

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e
            delay = 1
            print(f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
            time.sleep(delay)


def transcribe_dataset(
    dataset_path,
    dataset,
    split,
    model_name,
    use_url=False,
    max_samples=None,
    max_workers=4,
    mode="async",
    language="en"
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

    print(f"Transcribing with model: {model_name} in {mode} mode")

    def process_sample(sample):
        if use_url:
            reference = sample["row"]["text"].strip() or " "
            audio_duration = sample["row"]["audio_length_s"]
            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, None, sample, use_url=True, mode=mode, language=language
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
                    model_name, tmp_path, sample, use_url=False, mode=mode, language=language
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                os.unlink(tmp_path)
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--use_url", action="store_true")
    parser.add_argument("--mode", type=str, default="async", choices=["async", "realtime"])
    parser.add_argument("--language", type=str, default="en")

    args = parser.parse_args()

    model_name = args.model_name
    if args.mode == "async":
        model_name = "stt-async-preview"
    elif args.mode == "realtime":
        model_name = "stt-rt-preview-v2"

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=model_name,
        use_url=args.use_url,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        mode=args.mode,
        language=args.language,
    )
