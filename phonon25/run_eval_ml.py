"""Open ASR Leaderboard multilingual evaluation (FLEURS / MCV / MLS) for FermionResearch/Phonon-2.5.

Same model path as `run_eval.py` in this folder: the published `phonon-cuda` container image loads and verifies the
release file and builds the stock Transformers `ParakeetForTDT` graph; this script drives it batched, saves raw
predictions through `normalizer/data_utils.write_manifest`, and applies the leaderboard's multilingual normaliser only
for the printed WER (scoring is `score_results`). Dataset handling follows `nemo_asr/run_eval_ml.py`.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_AUDIO_DECODER_BACKEND", "soundfile")

import evaluate
import numpy as np
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs

wer_metric = evaluate.load("wer")

BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
IMAGE_MODULES = os.environ.get("PHONON_CUDA_MODULES", "/opt/phonon")


def load_phonon(model_id: str, dtype: str):
    sys.path.insert(0, IMAGE_MODULES)
    os.environ["PHONON2_CUDA_DTYPE"] = dtype
    os.environ.setdefault("PHONON2_CUDA_WARMUP", "first")
    import entrypoint as image
    from phonon2_cuda_engine import Transcriber
    key = image.resolve_model_name(model_id)
    key, model_dir = image.resolve_model(argparse.Namespace(model=key, repo=None, model_dir=None))
    engine = Transcriber(key, model_dir, image.CATALOG[key])
    return engine, engine.model


def main(args):
    torch.manual_seed(42)
    torch_dtype = getattr(torch, args.dtype)
    config_name = args.config_name
    language = args.language or (config_name.split("_", 1)[1] if "_" in config_name else "en")
    print(f"Detected language: {language}")

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

    print(f"Loading dataset: {args.dataset} with config: {config_name}")
    dataset = load_dataset(args.dataset, config_name, split=args.split, streaming=args.streaming)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    # Sort by duration (longest first) for efficient batching, as the NeMo multilingual script does.
    durations = list(dataset["duration"]) if "duration" in dataset.column_names else None
    if durations is not None and all(d is not None for d in durations):
        order = sorted(range(len(durations)), key=lambda i: durations[i], reverse=True)
        dataset = dataset.select(order)

    gen_kwargs = {}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    def benchmark(batch):
        audios = [np.asarray(a["array"], dtype=np.float32) for a in batch["audio"]]
        n = len(audios)
        batch["audio_length_s"] = [len(a) / sampling_rate for a in audios]
        start = time.time()
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
        runtime = time.time() - start
        batch["transcription_time_s"] = n * [runtime / n]
        batch["predictions"] = pred_text                      # raw; normalisation at scoring time
        batch["references"] = [data_utils.get_text(s) if isinstance(s, dict) else s for s in batch["text"]] \
            if "text" in batch else batch["sentence"]
        return batch

    if args.warmup_steps:
        n = args.warmup_steps * args.batch_size
        warm = dataset.select(range(min(n, len(dataset))))
        for _ in tqdm(iter(warm.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])),
                      desc="Warming up..."):
            continue

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    references, predictions, lengths, times = [], [], [], []
    for row in tqdm(iter(dataset), desc="Samples..."):
        if not data_utils.is_target_text_in_range(row["references"]):
            continue
        references.append(row["references"]); predictions.append(row["predictions"])
        lengths.append(row["audio_length_s"]); times.append(row["transcription_time_s"])

    manifest_path = data_utils.write_manifest(
        references, predictions, args.model_id, args.dataset, config_name, args.split,
        audio_length=lengths, transcription_time=times)
    print("Results saved at path:", os.path.abspath(manifest_path))

    if language == "en":
        norm_refs = [data_utils.normalizer(r) for r in references]
        norm_preds = [data_utils.normalizer(p) for p in predictions]
    else:
        norm_refs = [data_utils.ml_normalizer(r, lang=language) for r in references]
        norm_preds = [data_utils.ml_normalizer(p, lang=language) for p in predictions]
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = round(100 * wer_metric.compute(references=wer_refs, predictions=wer_preds), 2)
    rtfx = round(sum(lengths) / sum(times), 2)
    print(f"Dataset: {args.dataset}\nLanguage: {language}\nConfig: {config_name}\nModel: {args.model_id}")
    print(f"RTFX: {rtfx}")
    print(f"WER: {wer}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hf-audio/open-asr-leaderboard-multilingual-datasets")
    parser.add_argument("--config_name", type=str, required=True, help="<dataset>_<lang>, e.g. fleurs_de, mcv_fr, mls_es")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    main(args)
