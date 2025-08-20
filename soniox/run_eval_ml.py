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
                audio_url = sample["audio"][0]["src"]
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
                    response = requests.get(sample["audio"][0]["src"])
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
    config_name,
    split,
    model_name,
    max_samples=None,
    max_workers=4,
    mode="async",
    language="en"
):
    ds = datasets.load_dataset(dataset_path, config_name, split=split, streaming=False)
    # Disable automatic audio decoding to avoid torchcodec dependency
    ds = ds.cast_column('audio', datasets.Audio(decode=False))
    if max_samples:
        ds = ds.take(max_samples)

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    print(f"Transcribing with model: {model_name} in {mode} mode for language {language}")

    def process_sample(sample):
        reference = sample.get("text", "").strip() or " "
        
        # Handle undecoded audio - read from bytes
        audio_bytes = sample["audio"]["bytes"]
        
        # Write bytes to temporary file and read audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_bytes)
            tmpfile.flush()
            
            # Read back to get audio data and sample rate
            audio_data, sample_rate = sf.read(tmpfile.name)
            audio_duration = len(audio_data) / sample_rate
            
            tmp_path = tmpfile.name

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

    if language == "en":
        results["predictions"] = [data_utils.normalizer(pred) or " " for pred in results["predictions"]]
        results["references"] = [data_utils.normalizer(ref) or " " for ref in results["references"]]
    else:
        results["predictions"] = [data_utils.ml_normalizer(pred) or " " for pred in results["predictions"]]
        results["references"] = [data_utils.ml_normalizer(ref) or " " for ref in results["references"]]


    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        model_name.replace("/", "-"),
        dataset_path,
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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="async", choices=["async", "realtime"])

    args = parser.parse_args()

    model_name = args.model_name
    if args.mode == "async":
        model_name = "stt-async-preview"
    elif args.mode == "realtime":
        model_name = "stt-rt-preview-v2"

    transcribe_dataset(
        dataset_path=args.dataset,
        config_name=args.config_name,
        split=args.split,
        model_name=model_name,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        mode=args.mode,
        language=args.language,
    )
