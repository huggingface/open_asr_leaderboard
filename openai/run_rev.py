import argparse
import os
import time
import requests
from rev_ai import apiclient
from tqdm import tqdm
from dotenv import load_dotenv
import evaluate
import concurrent.futures
from normalizer import data_utils

load_dotenv()

REVAI_API_KEY = os.getenv("REVAI_API_KEY", "YOUR_DEFAULT_API_KEY")


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

class CustomerUrlData:
    def __init__(self, url: str):
        self.url = url
        
    def to_dict(self):
        return {"url": self.url}

def transcribe_with_rev(model_name, row, client, max_retries=10):
    retries = 0
    
    # Check audio length first
    if row['row']['audio_length_s'] < 2:
        return ".", row['row']['text'], row['row']['audio_length_s'], 0
    
    while retries <= max_retries:
        start = time.time()
        try:
            job = client.submit_job_url(
                transcriber=model_name.split("/")[1],
                source_config=CustomerUrlData(row['row']['audio'][0]['src']),
                metadata="benchmarking_job",
            )

            while True:
                job_details = client.get_job_details(job.id)
                if job_details.status.name in ["IN_PROGRESS", "TRANSCRIBING"]:
                    time.sleep(0.5)
                    continue
                elif job_details.status.name == "FAILED":
                    raise Exception("RevAI transcription failed.")
                elif job_details.status.name == "TRANSCRIBED":
                    break

            transcript_object = client.get_transcript_object(job.id)

            transcript_text = "".join(
                element.value
                for monologue in transcript_object.monologues
                for element in monologue.elements
            )

            transcription_time = time.time() - start
            return transcript_text, row['row']['text'], row['row']['audio_length_s'], transcription_time

        except Exception as e:
            retries += 1
            print(f"Error: {e}, retrying ({retries}/{max_retries})...")
            time.sleep(0.5)
    print(row['row']['audio_length_s'])
    return ".", row['row']['text'], row['row']['audio_length_s'], 0


def main(model_name, dataset_path, dataset, split, max_samples=None, num_jobs=4):
    client = apiclient.RevAiAPIClient(REVAI_API_KEY)
    audio_rows = fetch_audio_urls(dataset_path, dataset, split)
    predictions = []
    references = []
    audio_lengths = []
    transcription_times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = {executor.submit(transcribe_with_rev, model_name, row, client): row for row in audio_rows}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Transcribing audio"):
            transcript, reference, audio_length, transcription_time = future.result()
            if transcript:
                predictions.append(transcript)
                references.append(reference)
                audio_lengths.append(audio_length)
                transcription_times.append(transcription_time)
    
    predictions = [data_utils.normalizer(prediction) for prediction in predictions] 
    references = [data_utils.normalizer(reference) for reference in references]

    results = {
        "references": references,
        "predictions": predictions,
        "audio_length_s": audio_lengths,
        "transcription_time_s": transcription_times,
    }
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
    parser = argparse.ArgumentParser(description="Rev AI transcription script")
    parser.add_argument("--model_name", default="rev/machine")
    parser.add_argument("--dataset_path", default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", default="ami")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_jobs", type=int, default=100, help="Number of concurrent transcription jobs")

    args = parser.parse_args()

    main(args.model_name, args.dataset_path, args.dataset, args.split, args.max_samples, args.num_jobs)