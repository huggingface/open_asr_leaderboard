import argparse
import datasets
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from io import BytesIO
import assemblyai as aai
import openai
from elevenlabs.client import ElevenLabs
from rev_ai import apiclient
from rev_ai.models import CustomVocabulary, CustomerUrlData
from normalizer import data_utils
import concurrent.futures

load_dotenv()

def fetch_audio_urls(dataset_path, dataset, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={dataset}&split={split}"
    size_response = requests.get(size_url).json()
    total_rows = size_response['size']['config']['num_rows']
    audio_urls = []
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_rows - offset)
        }

        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                audio_urls.extend(data['rows'])
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                retries += 1
                print(f"Error fetching data: {e}, retrying ({retries}/{max_retries})...")
                time.sleep(10)
                if retries >= max_retries:
                    raise Exception("Max retries exceeded while fetching data.")
    time.sleep(1)
    return audio_urls

def transcribe_with_retry(model_name, audio_file_path, sample, max_retries=10, use_url=False):
    retries = 0
    while retries <= max_retries:
        try:
            if model_name.startswith("assembly/"):
                aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(
                    speech_model=model_name.split("/")[1],
                    language_code="en",
                )
                if use_url:
                    audio_url = sample['row']['audio'][0]['src']
                    audio_duration = sample['row']['audio_length_s']
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    transcript = transcriber.transcribe(audio_url, config=config)
                else:
                    audio_duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                    if audio_duration < 0.160:
                        print(f"Skipping audio duration {audio_duration}s")
                        return "."
                    transcript = transcriber.transcribe(audio_file_path, config=config)
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"AssemblyAI transcription error: {transcript.error}")
                return transcript.text

            elif model_name.startswith("openai/"):
                if use_url:
                    response = requests.get(sample['row']['audio'][0]['src'])
                    audio_data = BytesIO(response.content)
                    response = openai.Audio.transcribe(
                        model=model_name.split("/")[1],
                        file=audio_data,
                        response_format="text",
                        language="en",
                        temperature=0.0,
                    )
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        response = openai.Audio.transcribe(
                            model=model_name.split("/")[1],
                            file=audio_file,
                            response_format="text",
                            language="en",
                            temperature=0.0,
                        )
                return response.strip()

            elif model_name.startswith("elevenlabs/"):
                client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                if use_url:
                    response = requests.get(sample['row']['audio'][0]['src'])
                    audio_data = BytesIO(response.content)
                    transcription = client.speech_to_text.convert(
                        file=audio_data,
                        model_id=model_name.split("/")[1],
                        language_code="eng",
                        tag_audio_events=True,

                    )
                else:
                    with open(audio_file_path, "rb") as audio_file:
                        transcription = client.speech_to_text.convert(
                            file=audio_file,
                            model_id=model_name.split("/")[1],
                            language_code="eng",
                            tag_audio_events=True,
                        )
                return transcription.text
            
            elif model_name.startswith("revai/"):
                access_token = os.getenv("REVAI_API_KEY")
                client = apiclient.RevAiAPIClient(access_token)

                if use_url:
                    # Submit job with URL for Rev.ai
                    job = client.submit_job_url(
                        transcriber=model_name.split("/")[1],
                        source_config=CustomerUrlData(sample['row']['audio'][0]['src']),
                        metadata="benchmarking_job",
                    )
                else:
                    # Submit job with local file
                    job = client.submit_job_local_file(
                        transcriber=model_name.split("/")[1],
                        filename=audio_file_path,
                        metadata="benchmarking_job",
                    )

                # Polling until job is done
                while True:
                    job_details = client.get_job_details(job.id)
                    if job_details.status.name in ["IN_PROGRESS", "TRANSCRIBING"]:
                        time.sleep(0.1)
                        continue
                    elif job_details.status.name == "FAILED":
                        raise Exception("RevAI transcription failed.")
                    elif job_details.status.name == "TRANSCRIBED":
                        break

                transcript_object = client.get_transcript_object(job.id)
                
                # Combine all words from all monologues
                transcript_text = []
                for monologue in transcript_object.monologues:
                    for element in monologue.elements:
                        transcript_text.append(element.value)

                return "".join(transcript_text) if transcript_text else ""

            else:
                raise ValueError("Invalid model prefix, must start with 'assembly/', 'openai/', 'elevenlabs/' or 'revai/'")

        except Exception as e:
            retries += 1
            if retries > max_retries:
                return "."

            if not use_url:
                sf.write(audio_file_path, sample["audio"]["array"], sample["audio"]["sampling_rate"], format="WAV")
            delay = 1
            print(f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
            time.sleep(delay)


def transcribe_dataset(dataset_path, dataset, split, model_name, use_url=False, max_samples=None, max_workers=4):
    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, dataset, split)
        if max_samples:
            audio_rows = audio_rows[:max_samples]
        ds = audio_rows
    else:
        ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)
        ds = data_utils.prepare_data(ds)
        if max_samples:
            ds = ds.take(max_samples)

    results = {"references": [], "predictions": [], "audio_length_s": [], "transcription_time_s": []}

    print(f"Transcribing with model: {model_name}")

    def process_sample(sample):
        if use_url:
            reference = sample['row']['text'].strip() or " "
            audio_duration = sample['row']['audio_length_s']
            start = time.time()
            try:
                transcription = transcribe_with_retry(model_name, None, sample, use_url=True)
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                return None

        else:
            reference = sample.get("norm_text", "").strip() or " "
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, sample["audio"]["array"], sample["audio"]["sampling_rate"], format="WAV")
                tmp_path = tmpfile.name
                audio_duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]

            start = time.time()
            try:
                transcription = transcribe_with_retry(model_name, tmp_path, sample, use_url=False)
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
        future_to_sample = {executor.submit(process_sample, sample): sample for sample in ds}
        for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(future_to_sample), desc="Transcribing"):
            result = future.result()
            if result:
                reference, transcription, audio_duration, transcription_time = result
                results["predictions"].append(transcription)
                results["references"].append(reference)
                results["audio_length_s"].append(audio_duration)
                results["transcription_time_s"].append(transcription_time)

    results["predictions"] = [data_utils.normalizer(transcription) or " " for transcription in results["predictions"]]
    results["references"] = [data_utils.normalizer(reference) or " " for reference in results["references"]]

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
    wer = wer_metric.compute(references=results["references"], predictions=results["predictions"])
    wer_percent = round(100 * wer, 2)
    rtfx = round(sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2)

    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Transcription Script with Concurrency")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name", required=True, help="Prefix model name with 'assembly/', 'openai/', or 'elevenlabs/'")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=300, help="Number of concurrent threads")
    parser.add_argument("--use_url", action="store_true", help="Use URL-based audio fetching instead of datasets")

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
