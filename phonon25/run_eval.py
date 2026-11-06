"""Open ASR Leaderboard evaluation (English short-form) for the Phonon-2.5 family.

Models: FermionResearch/Phonon-2.5 and FermionResearch/Phonon-2.5-English (also FermionResearch/Phonon-2).

The model is loaded by the published `phonon-cuda` container image (`ghcr.io/fermionresearch/phonon-cuda`), which
downloads and verifies the release file and builds the stock Transformers `ParakeetForTDT` graph from it. This script
only drives that graph the way this repository's `transformers/run_eval.py` does: batched `generate`, data loading and
manifest writing through `normalizer/data_utils.py`, raw predictions saved, normalisation at scoring time.

Two decode settings come from the image's model catalog and are applied here exactly as `phonon-cuda transcribe` and the
pip package apply them: a short leading pad of digital silence per clip, and the Greek word-final-sigma spelling rule
(both zero/no-op for Phonon-2 and Phonon-2.5 English).
"""
import argparse
import os
import random
import sys
import time

import evaluate
import numpy as np
import torch
from tqdm import tqdm

from normalizer import data_utils

wer_metric = evaluate.load("wer")

BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"  # processor + generation config; the weights come from the Phonon release file
IMAGE_MODULES = os.environ.get("PHONON_CUDA_MODULES", "/opt/phonon")


def load_phonon(model_id: str, dtype: str):
    """-> (image Transcriber, Transformers ParakeetForTDT on CUDA). Uses the container image's own loader."""
    sys.path.insert(0, IMAGE_MODULES)
    os.environ["PHONON2_CUDA_DTYPE"] = dtype
    os.environ.setdefault("PHONON2_CUDA_WARMUP", "first")  # the eval does its own warm-up
    import entrypoint as image                      # catalog, download, verification (the image's CLI module)
    from phonon2_cuda_engine import Transcriber     # the image's engine
    key = image.resolve_model_name(model_id)
    key, model_dir = image.resolve_model(argparse.Namespace(model=key, repo=None, model_dir=None))
    engine = Transcriber(key, model_dir, image.CATALOG[key])
    return engine, engine.model


def main(args):
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch_dtype = getattr(torch, args.dtype)

    from transformers import AutoProcessor, GenerationConfig

    engine, model = load_phonon(args.model_id, args.dtype)
    model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
    model.eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    sampling_rate = processor.feature_extractor.sampling_rate

    lead_pad = int(round(float(engine.lead_pad_s or 0.0) * sampling_rate))
    pad = np.zeros(lead_pad, dtype=np.float32)
    apply_sigma = bool(getattr(engine, "final_sigma", False))
    if apply_sigma:
        from phonon2_cuda_engine import final_sigma
    print(f"decode settings from the image catalog: lead pad {lead_pad} samples, final-sigma rule {apply_sigma}")

    gen_kwargs = {}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    def benchmark(batch, min_new_tokens=None):
        audios = [np.asarray(audio["array"], dtype=np.float32) for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(a) / sampling_rate for a in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)
        start_time = time.time()
        if lead_pad:
            audios = [np.concatenate([pad, a]) for a in audios]
        inputs = processor(audios, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: (v.to(args.device, torch_dtype) if v.dtype.is_floating_point else v.to(args.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        pred_text = processor.batch_decode(getattr(out, "sequences", out), skip_special_tokens=True)
        if apply_sigma:
            pred_text = [final_sigma(t) for t in pred_text]
        runtime = time.time() - start_time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text            # raw; normalisation at scoring time
        batch["references"] = batch["original_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.prepare_data(data_utils.load_data(args), sampling_rate=sampling_rate)
        n = args.warmup_steps * args.batch_size
        warm = dataset.take(n) if args.streaming else dataset.select(range(min(n, len(dataset))))
        for _ in tqdm(iter(warm.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])),
                      desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        dataset = dataset.take(args.max_eval_samples) if args.streaming \
            else dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate)
    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    # Chunked sets (Earnings-22 chunked) carry one reference per parent session: keep parent_id / chunk_index so
    # score_results can reassemble the chunks (this repository's transformers/run_eval.py does the same).
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    all_results = {"audio_length_s": [], "transcription_time_s": [], "predictions": [], "references": [],
                   "audio_filepath": []}
    if is_chunked:
        all_results.update({key: [] for key in data_utils.CHUNK_METADATA_KEYS})
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"], args.model_id, args.dataset_path,
        args.dataset or "", args.split,
        audio_length=all_results["audio_length_s"], transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
        extra_fields={key: all_results[key] for key in data_utils.CHUNK_METADATA_KEYS} if is_chunked else None)
    print("Results saved at path:", os.path.abspath(manifest_path))
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    if is_chunked:
        print("chunked dataset: per-chunk WER is not meaningful; WER comes from score_results after chunk merge.",
              "RTFx:", rtfx)
    else:
        norm_refs = [data_utils.normalizer(r) for r in all_results["references"]]
        norm_preds = [data_utils.normalizer(p) for p in all_results["predictions"]]
        wer = round(100 * wer_metric.compute(references=norm_refs, predictions=norm_preds), 2)
        print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="FermionResearch/Phonon-2.5, FermionResearch/Phonon-2.5-English or FermionResearch/Phonon-2 "
                             "(or the image aliases phonon-2.5 / phonon-2.5-en / phonon-2)")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", type=str, default="", help="Dataset config name (empty for single-config repos).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (the image requires a GPU).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--revision", type=str, default=None, help=argparse.SUPPRESS)
    parser.set_defaults(streaming=False)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    main(args)
