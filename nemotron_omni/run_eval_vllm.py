"""Open ASR Leaderboard runner for Nemotron-3-Nano-Omni via vLLM's OpenAI server.

Start the server first with `./run_server.sh`, then invoke this script. Each
sample is dumped to a short-lived WAV file in a temp dir and referenced by a
`file://` URL in the chat completion request, matching the model-card example.
Concurrency is provided by an in-process thread pool against the same server.
"""

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import evaluate
import numpy as np
import soundfile as sf
from openai import OpenAI
from tqdm import tqdm

from normalizer import data_utils


wer_metric = evaluate.load("wer")

SAMPLE_RATE = 16_000


def transcribe_one(client, model_id, audio_array, user_prompt, max_tokens,
                   seed, tmpdir, idx):
    wav_path = Path(tmpdir) / f"sample_{os.getpid()}_{idx}.wav"
    sf.write(str(wav_path), audio_array, SAMPLE_RATE)
    audio_url = wav_path.resolve().as_uri()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
            seed=seed,
            extra_body={
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content or ""
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass


def main(args):
    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")
    tmpdir = tempfile.mkdtemp(prefix="nemotron_omni_audio_")

    def benchmark(batch):
        audios = [np.asarray(a["array"], dtype=np.float32) for a in batch["audio"]]
        n = len(audios)
        batch["audio_length_s"] = [len(a) / SAMPLE_RATE for a in audios]

        start = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    transcribe_one,
                    client,
                    args.model_id,
                    audio,
                    args.user_prompt,
                    args.max_new_tokens,
                    args.seed,
                    tmpdir,
                    i,
                )
                for i, audio in enumerate(audios)
            ]
            raw_texts = [f.result() for f in futures]
        runtime = time.time() - start
        per_sample_runtime = runtime / max(1, n)

        batch["transcription_time_s"] = [per_sample_runtime] * n
        batch["predictions"] = [data_utils.normalizer(t) for t in raw_texts]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(
                range(min(num_warmup_samples, len(warmup_dataset)))
            )
        warmup_dataset = iter(
            warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True)
        )
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"],
        predictions=all_results["predictions"],
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]),
        2,
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="Transcribe the audio clip into text. Return only the transcription.",
    )
    parser.set_defaults(streaming=False)
    args = parser.parse_args()
    main(args)
