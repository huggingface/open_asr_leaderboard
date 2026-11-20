"""Multilingual ASR evaluation for models that run in the Cactus engine (`pip install cactus-compute`).

Written for udgamlabs/Shruti (https://huggingface.co/udgamlabs/Shruti): speech-to-text for Hindi, English and
Hinglish, 55M parameters, fine-tuned from Cactus Whistle. The model ships as one 4-bit `.cact` file (29.5 MB) that runs
in the Cactus engine on CPU (Apache-2.0); there is no GPU path, so this script runs on the job's CPU and `--device` is
ignored. Results are what users of the released file get.

Decoding: the engine's defaults (its built-in beam search), no prompt, no keywords. The language is detected
automatically by the model (`--force_language` exists but is off by default and is not used for the leaderboard), so
`--language` only selects the normalizer/scorer.

Batching: the engine loads one model per process and is not thread-safe, so each batch is split across
`--num_workers` worker processes, each holding its own copy of the model and running one engine thread (default:
one worker per CPU the container may use). Batches run in
dataset order like the other scripts; RTFx is the wall time of each batch.
"""

import argparse
import json
import os
import re
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("CACTUS_TELEMETRY", "0")  # no usage pings from the benchmark job
# One math-library thread per worker process: parallelism comes from the workers, and a container can report far more
# host cores than it may use (OpenBLAS then fails to start its threads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import evaluate
import numpy as np
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs

wer_metric = evaluate.load("wer")

SAMPLING_RATE = 16_000

# Model id -> (weights file in the repo, parameter count). The .cact file holds 4-bit weights, so the count comes from
# the model card (8 encoder + 8 decoder layers, width 512), not from the file.
MODELS = {
    "udgamlabs/Shruti": ("shruti.cact", 55_000_000),
}

_engine = None
_force_language = None


def usable_cpus():
    """CPUs this container may use: the CPU set, capped by a cgroup v2 quota (os.cpu_count() reports the host)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def _init_worker(weights, force_language):
    global _engine, _force_language
    from cactus import Whistle
    from cactus.agent.runtime import set_threads

    set_threads(1)  # one engine thread per worker process (the automatic count would use every host core)
    _engine = Whistle(weights)
    _force_language = force_language


def _transcribe(audio):
    out = _engine.transcribe(np.asarray(audio, dtype=np.float32), language=_force_language)
    return out.get("text", "")


def main(args):
    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split

    if args.language:
        LANGUAGE = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        LANGUAGE = lang_match.group(1) if lang_match else "hi"

    weights_file, n_params = MODELS[args.model_id]
    weights = hf_hub_download(args.model_id, weights_file, revision=args.revision)
    print(f"Model size: {n_params / 1e9:.3f}B parameters ({os.path.getsize(weights) / 1e6:.1f} MB weights file)")

    workers = max(1, args.num_workers or usable_cpus())
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(weights, args.force_language))
    list(pool.map(_transcribe, [np.zeros(SAMPLING_RATE, np.float32)] * workers))  # start workers, load the model
    print(f"{workers} worker processes, one engine each, on CPU")

    def load_eval_dataset():
        dataset = load_dataset(args.dataset, CONFIG_NAME, split=SPLIT_NAME, streaming=args.streaming, token=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            if args.streaming:
                dataset = dataset.take(args.max_eval_samples)
            else:
                dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
        return dataset

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        start_time = time.perf_counter()
        pred_text = list(pool.map(_transcribe, audios))
        runtime = time.perf_counter() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        if "lattice" in batch:
            batch["references"] = [json.dumps(lat, ensure_ascii=False) for lat in batch["lattice"]]
        else:
            batch["references"] = batch["text"]
        return batch

    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_eval_dataset()

    if args.warmup_steps is not None and args.warmup_steps > 0:
        # Warm-up on synthetic audio only (no test-set information): quiet noise at a few fixed lengths.
        print(f"Running {args.warmup_steps} warmup steps on synthetic audio...")
        rng = np.random.default_rng(0)
        for _ in range(args.warmup_steps):
            fake = [(0.01 * rng.standard_normal(int(SAMPLING_RATE * s))).astype(np.float32)
                    for s in [1, 5, 10, 20] * ((workers + 3) // 4)]
            list(pool.map(_transcribe, fake))

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    all_results = {k: [] for k in ("audio_length_s", "transcription_time_s", "predictions", "references",
                                   "audio_filepath")}
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    filtered = [
        row for row in zip(all_results["references"], all_results["predictions"], all_results["audio_length_s"],
                           all_results["transcription_time_s"], all_results["audio_filepath"])
        if data_utils.is_target_text_in_range(row[0])
    ]
    if filtered:
        (all_results["references"], all_results["predictions"], all_results["audio_length_s"],
         all_results["transcription_time_s"], all_results["audio_filepath"]) = zip(*filtered)
        all_results = {k: list(v) for k, v in all_results.items()}

    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"], args.model_id, args.dataset, CONFIG_NAME or "",
        args.split, audio_length=all_results["audio_length_s"], transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    from normalizer.eval_utils import OIWER_LANGUAGES, score_oiwer

    if LANGUAGE in OIWER_LANGUAGES:
        manifest = [{"text": ref, "pred_text": pred}
                    for ref, pred in zip(all_results["references"], all_results["predictions"])]
        wer, _ins, _del, _sub = score_oiwer(manifest, OIWER_LANGUAGES[LANGUAGE])
        wer = round(100 * wer, 2)
    else:
        norm_refs = [data_utils.ml_normalizer(r, lang=LANGUAGE) for r in all_results["references"]]
        norm_preds = [data_utils.ml_normalizer(p, lang=LANGUAGE) for p in all_results["predictions"]]
        wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
        wer = round(100 * wer_metric.compute(references=wer_refs, predictions=wer_preds), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)
    pool.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="udgamlabs/Shruti", help="One of MODELS.")
    parser.add_argument("--revision", type=str, default="main", help="Model repo revision (commit or tag).")
    parser.add_argument("--dataset", type=str, required=True, help="E.g. 'VoiceArena/Monsoon_hi_test'.")
    parser.add_argument("--config_name", type=str, default=None, help="Omit for single-config dataset repos.")
    parser.add_argument("--language", type=str, default=None, help="Selects the normalizer/scorer only.")
    parser.add_argument("--force_language", type=str, default=None,
                        help="Force the decoding language (off: detected automatically). Not used for the leaderboard.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1, help="Ignored: the Cactus engine runs on CPU.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0, help="Engine instances (0: one per CPU core).")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=2)
    main(parser.parse_args())
