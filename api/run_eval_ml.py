import argparse
import concurrent.futures
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional

import datasets
from datasets import Audio
import evaluate
import soundfile as sf
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Make the repository-level normalizer importable when this entry point is
# launched directly with Windows Python (which uses ';', not ':', in
# PYTHONPATH). This is a no-op in the Linux container setup.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from normalizer import data_utils  # noqa: E402
from normalizer.eval_utils import normalize_compound_pairs  # noqa: E402
from providers import PermanentError, get_provider  # noqa: E402

load_dotenv()


def fetch_audio_urls(dataset_path, config_name, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={config_name}&split={split}"
    size_response = requests.get(size_url).json()
    total_rows = size_response["size"]["config"]["num_rows"]
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": config_name,
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
                response = requests.get(API_URL, params=params, headers=headers)
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
    language="en",
    prompt=None,
):
    provider, variant = get_provider(model_name)
    kwargs = dict(use_url=use_url, language=language)
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
    config_name,
    split,
    model_name,
    language,
    use_url=False,
    max_samples=None,
    max_workers=4,
    prompt=None,
    resume=False,
    seed_manifest=None,
):
    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, config_name, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = list(audio_rows)
    else:
        ds = datasets.load_dataset(dataset_path, config_name, split=split, streaming=False)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

    model_slug = model_name.replace("/", "-")
    dataset_slug = dataset_path.replace("/", "-")
    checkpoint_dir = Path("results") / ".checkpoints"
    checkpoint_path = checkpoint_dir / (
        f"MODEL_{model_slug}_DATASET_{dataset_slug}_{config_name}_{split}.jsonl.partial"
    )
    completed: dict[int, tuple[str, str, float, float]] = {}
    if resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
            for line in checkpoint_file:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print("Ignoring an incomplete trailing checkpoint record")
                    continue
                completed[int(item["index"])] = (
                    item["reference"],
                    item["prediction"],
                    float(item["audio_length_s"]),
                    float(item["transcription_time_s"]),
                )
        print(f"Resuming from {len(completed)} checkpointed samples")

    if resume and seed_manifest:
        seed_path = Path(seed_manifest)
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed manifest does not exist: {seed_path}")
        seeded = 0
        with seed_path.open("r", encoding="utf-8") as seed_file:
            for index, line in enumerate(seed_file):
                if not line.strip() or index in completed:
                    continue
                if index >= len(ds):
                    raise ValueError(
                        f"Seed manifest has more rows than dataset {config_name}"
                    )
                item = json.loads(line)
                completed[index] = (
                    str(item["text"]),
                    str(item["pred_text"]),
                    float(item["duration"]),
                    float(item["time"]),
                )
                seeded += 1
        print(f"Seeded {seeded} completed samples from {seed_path}")

    print(f"Transcribing with model: {model_name}, language: {language}, config: {config_name}")

    def process_sample(index):
        sample = ds[index]
        if use_url:
            reference = sample["row"]["text"].strip()
            audio_duration = sample["row"]["audio_length_s"]
            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, None, sample, use_url=True, language=language, prompt=prompt
                )
            except Exception as e:
                return index, None, str(e)

        else:
            reference = sample.get("text", "").strip()
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
                    model_name, tmp_path, sample, use_url=False, language=language, prompt=prompt
                )
            except Exception as e:
                return index, None, str(e)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        transcription_time = time.time() - start
        return index, (
            reference,
            transcription,
            audio_duration,
            transcription_time,
        ), None

    sample_count = len(ds)
    pending_indices = [index for index in range(sample_count) if index not in completed]
    failures = []
    checkpoint_file = None
    if resume:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path.open("a", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_sample, index): index for index in pending_indices
        }
        try:
            for future in tqdm(
                concurrent.futures.as_completed(future_to_index),
                total=len(future_to_index),
                desc="Transcribing",
            ):
                index, result, error = future.result()
                if result is None:
                    failures.append((index, error))
                    print(f"Sample {index} failed after retries: {error}")
                    continue
                completed[index] = result
                if checkpoint_file is not None:
                    reference, prediction, audio_duration, transcription_time = result
                    checkpoint_file.write(
                        json.dumps(
                            {
                                "index": index,
                                "reference": reference,
                                "prediction": prediction,
                                "audio_length_s": audio_duration,
                                "transcription_time_s": transcription_time,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    checkpoint_file.flush()
        finally:
            if checkpoint_file is not None:
                checkpoint_file.close()

    if failures:
        preview = ", ".join(str(index) for index, _ in failures[:10])
        raise RuntimeError(
            f"{len(failures)} samples failed (indices: {preview}). "
            f"Successful samples remain checkpointed at {checkpoint_path}."
        )

    ordered_results = [completed[index] for index in range(sample_count)]
    results = {
        "references": [item[0] for item in ordered_results],
        "predictions": [item[1] for item in ordered_results],
        "audio_length_s": [item[2] for item in ordered_results],
        "transcription_time_s": [item[3] for item in ordered_results],
    }

    # Filter empty references (consistent with English pipeline's prepare_data)
    filtered = [
        (ref, pred, dur, time_s)
        for ref, pred, dur, time_s in zip(
            results["references"], results["predictions"],
            results["audio_length_s"], results["transcription_time_s"]
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        results["references"], results["predictions"], results["audio_length_s"], results["transcription_time_s"] = zip(*filtered)
        results = {k: list(v) for k, v in results.items()}

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

    norm_refs = [data_utils.ml_normalizer(r, lang=language) for r in results["references"]]
    norm_preds = [data_utils.ml_normalizer(t, lang=language) for t in results["predictions"]]
    wer_metric = evaluate.load("wer")
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
    wer_percent = round(100 * wer, 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)
    if resume and checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multilingual API Transcription Script with Concurrency"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--config_name", required=True, help="Dataset config name, e.g. 'fleurs_de'")
    parser.add_argument("--language", required=True, help="Language code, e.g. 'de'")
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
        "--use_url",
        action="store_true",
        help="Use URL-based audio fetching instead of datasets",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt to pass to the provider (e.g., 'Output must be in lexical format.')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Checkpoint successful samples and resume an interrupted evaluation.",
    )
    parser.add_argument(
        "--seed_manifest",
        default=None,
        help="Completed ordered manifest whose rows should seed a resumed run.",
    )

    args = parser.parse_args()

    transcribe_dataset(
        dataset_path=args.dataset_path,
        config_name=args.config_name,
        split=args.split,
        model_name=args.model_name,
        language=args.language,
        use_url=args.use_url,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        prompt=args.prompt,
        resume=args.resume,
        seed_manifest=args.seed_manifest,
    )
