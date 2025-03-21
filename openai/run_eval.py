import argparse
import datasets
import evaluate
import io
import json
import soundfile as sf
import tempfile
import time
from tqdm import tqdm
import openai
from normalizer import data_utils  # must provide .normalizer() and .write_manifest()

def transcribe_dataset(
    dataset_path, dataset, split,
    model_name="whisper-1",
):
    # Load dataset
    ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)

    # Track results
    all_results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    print(f"Transcribing with OpenAI model: {model_name}")

    for i, sample in tqdm(enumerate(ds), total=len(ds), desc="Transcribing"):
        # Get reference text, use empty string if not present
        reference = sample.get("text", "").strip()
        
        # Write temp .wav file
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, sample["audio"]["array"], sample["audio"]["sampling_rate"], format="WAV")

            start = time.time()
            response = openai.Audio.transcribe(
                model=model_name,
                file=tmpfile,
                response_format="text"
            )
            end = time.time()

        transcription = response.strip()
        reference = sample["text"]
        audio_duration = sample["audio_length_s"]
        transcription_time = end - start

        transcription = data_utils.normalizer(transcription)
        reference = data_utils.normalizer(reference)
        # Store
        all_results["predictions"].append(transcription)
        all_results["references"].append(reference)
        all_results["audio_length_s"].append(audio_duration)
        all_results["transcription_time_s"].append(transcription_time)

    # Save results to manifest
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        model_name,
        dataset_path,
        dataset,
        split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", manifest_path)

    # Evaluate
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(
        references=all_results["references"],
        predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]),
        2
    )

    print("WER:", wer, "%", "RTFx:", rtfx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe using OpenAI Whisper API")

    parser.add_argument("--dataset_path", required=True, help="Dataset path or name")
    parser.add_argument("--dataset", required=True, help="Subset name of the dataset")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--model_name", default="whisper-1", help="OpenAI model name")

    args = parser.parse_args()

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
    )
