import argparse
import concurrent.futures
import os
import tempfile
import time

import evaluate
import soundfile as sf
from dotenv import load_dotenv
from gladiaio_sdk import GladiaClient
from tqdm import tqdm

from normalizer import data_utils

load_dotenv()

wer_metric = evaluate.load("wer")


class GladiaTranscriber:
    """Gladia Solaria pre-recorded transcription via the official SDK."""

    def __init__(self, language: str = "en", region: str | None = None):
        api_key = os.environ.get("GLADIA_API_KEY")
        if not api_key:
            raise ValueError(
                "GLADIA_API_KEY is not set. Export it or add it to a .env file."
            )
        kwargs = {"api_key": api_key}
        if region:
            kwargs["region"] = region
        self.client = GladiaClient(**kwargs)
        self.language = language
        self._warmed_up = False

    def _transcription_options(self) -> dict:
        return {"language_config": {"languages": [self.language]}}

    def transcribe_file(self, audio_path: str) -> str:
        response = self.client.prerecorded().transcribe(
            audio_path,
            self._transcription_options(),
        )
        if response.status != "done":
            raise RuntimeError(
                f"Gladia transcription failed with status={response.status!r}"
            )
        transcription = response.result and response.result.transcription
        if transcription is None or not transcription.full_transcript:
            return " "
        return transcription.full_transcript.strip()

    def warmup(self, audio_path: str) -> None:
        if self._warmed_up:
            return
        self.transcribe_file(audio_path)
        self._warmed_up = True


def transcribe_with_retry(
    transcriber: GladiaTranscriber,
    audio_path: str,
    max_retries: int = 10,
) -> str:
    retries = 0
    while retries <= max_retries:
        try:
            return transcriber.transcribe_file(audio_path)
        except Exception as exc:
            retries += 1
            if retries > max_retries:
                raise exc
            delay = min(2**retries, 30)
            print(
                f"Gladia API error: {exc}. Retrying in {delay}s "
                f"(attempt {retries}/{max_retries})..."
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable")


def process_sample(transcriber: GladiaTranscriber, sample: dict):
    reference = sample.get("norm_text", "").strip() or " "
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(
            tmpfile.name,
            sample["audio"]["array"],
            sample["audio"]["sampling_rate"],
            format="WAV",
        )
        tmp_path = tmpfile.name
        audio_duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]

    start = time.time()
    try:
        transcription = transcribe_with_retry(transcriber, tmp_path)
    except Exception as exc:
        print(f"Failed to transcribe sample after retries: {exc}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    transcription_time = time.time() - start
    return reference, transcription, audio_duration, transcription_time


def transcribe_dataset(args: argparse.Namespace) -> None:
    transcriber = GladiaTranscriber(language=args.language, region=args.region)

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(
                range(min(args.max_eval_samples, len(dataset)))
            )
    dataset = data_utils.prepare_data(dataset)

    if args.warmup_steps > 0:
        warmup_samples = list(
            dataset.take(args.warmup_steps)
            if args.streaming
            else dataset.select(range(min(args.warmup_steps, len(dataset))))
        )
        if warmup_samples:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sample = warmup_samples[0]
                sf.write(
                    tmpfile.name,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
                warmup_path = tmpfile.name
            try:
                print(f"Running {args.warmup_steps} warmup request(s)...")
                for _ in range(args.warmup_steps):
                    transcriber.warmup(warmup_path)
            finally:
                if os.path.exists(warmup_path):
                    os.unlink(warmup_path)

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    samples = list(dataset)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [
            executor.submit(process_sample, transcriber, sample)
            for sample in samples
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Transcribing",
        ):
            result = future.result()
            if result is None:
                continue
            reference, transcription, audio_duration, transcription_time = result
            results["references"].append(reference)
            results["predictions"].append(transcription)
            results["audio_length_s"].append(audio_duration)
            results["transcription_time_s"].append(transcription_time)

    results["predictions"] = [
        data_utils.normalizer(transcription) or " "
        for transcription in results["predictions"]
    ]
    results["references"] = [
        data_utils.normalizer(reference) or " "
        for reference in results["references"]
    ]

    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=results["references"],
        predictions=results["predictions"],
    )
    wer_percent = round(100 * wer, 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]),
        2,
    )
    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Gladia Solaria on the Open ASR Leaderboard"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="gladia/solaria-3",
        help="Model identifier used for result manifests.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/open-asr-leaderboard",
        help="Hugging Face dataset path.",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate. Use a small value (e.g. 64) for smoke tests.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Download the full dataset instead of streaming it.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Number of concurrent Gladia API requests.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="ISO 639-1 language code passed to Gladia.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Gladia region (e.g. eu-west, us-west). Defaults to GLADIA_REGION env var.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1,
        help="Number of warmup API calls before timed evaluation.",
    )
    parser.set_defaults(streaming=True)

    transcribe_dataset(parser.parse_args())
