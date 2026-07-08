import argparse
from typing import Optional
import datasets
import soundfile as sf
import tempfile
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from normalizer import data_utils
import concurrent.futures
from providers import get_provider, PermanentError

load_dotenv()


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    language="en",
    prompt=None,
):
    provider, variant = get_provider(model_name)
    kwargs = dict(language=language)
    if prompt is not None:
        kwargs["prompt"] = prompt
    retries = 0
    while retries <= max_retries:
        try:
            return provider.transcribe(variant, audio_file_path, sample, **kwargs)
        except PermanentError:
            raise
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

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
    max_samples=None,
    max_workers=4,
    prompt=None,
):
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
        reference = sample.get("original_text", "").strip() or " "
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
                model_name, tmp_path, sample, prompt=prompt
            )
        except Exception as e:
            print(f"Failed to transcribe after retries: {e}")
            transcription = ""
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
            reference, transcription, audio_duration, transcription_time = future.result()
            results["predictions"].append(transcription)
            results["references"].append(reference)
            results["audio_length_s"].append(audio_duration)
            results["transcription_time_s"].append(transcription_time)

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

    from kaldialign import batch_error_rate
    norm_refs = [data_utils.normalizer(r) for r in results["references"]]
    norm_preds = [data_utils.normalizer(p) for p in results["predictions"]]
    refs_split = [tuple(r.split()) for r in norm_refs]
    preds_split = [tuple(p.split()) for p in norm_preds]
    wer = round(100 * batch_error_rate(refs_split, preds_split, merge_compounds=True)["err_rate"], 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("WER:", wer, "%")
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
        help="Prefix model name with provider prefix (e.g., 'assembly/', 'openai/', 'elevenlabs/', 'revai/', 'speechmatics/' or 'aquavoice/')",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_workers", type=int, default=300, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt to pass to the provider (e.g., 'Output must be in lexical format.')",
    )

    args = parser.parse_args()

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        prompt=args.prompt,
    )