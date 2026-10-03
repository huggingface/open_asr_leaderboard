"""Open ASR Leaderboard evaluation for Phonon-2 (FermionResearch/Phonon-2 placeholder id).

The model is an under-2-bit Parakeet-TDT-0.6B encoder with int6 dense tables, shipped as a
`fermion-five-value-parakeet-v1` container. This script expands the container to a dense state dict (exact:
five-value -> {0, +-lo, +-hi}; int6 -> q*scale; fp16 as stored) and runs it through the stock `transformers`
ParakeetForTDT graph -- the same correctness path the model card's numbers were measured on. It follows the
Transformers reference `run_eval.py` of this repository: batch processing, `normalizer/data_utils.py` for data loading
and manifest writing, raw predictions saved, normalisation applied at scoring time.

Speed note: the packed (two 2-bit QMM) kernels exist for Apple MLX only; on CUDA this dense fp32/bf16 path is a
correctness reference and its RTFx is that of a dense 0.6B Parakeet-TDT in transformers, not of the packed runtime.
"""
import argparse
import os
import random
import time
from pathlib import Path

import evaluate
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from normalizer import data_utils


# --- Phonon-2 weight container (`fermion-five-value-parakeet-v1`) reader; self-contained -------------------------------
import json  # noqa: E402

FORMAT = "fermion-five-value-parakeet-v1"



def _trits(buf: bytes, o: int, i: int) -> np.ndarray:
    rb = (i + 4) // 5
    x = np.frombuffer(buf, dtype=np.uint8, count=o * rb).reshape(o, rb).astype(np.uint16)
    d = np.empty((o, rb, 5), dtype=np.uint8)
    for k in range(5):
        d[:, :, k] = (x // (3 ** k)) % 3
    return d.reshape(o, rb * 5)[:, :i]


def _five_value(blob: bytes, shape, raw: dict | None = None) -> np.ndarray:
    o, i = shape
    rb = (i + 4) // 5
    codes = _trits(blob[: o * rb], o, i)
    nzmask = codes != 1
    nnz = int(nzmask.sum())
    rbytes = (nnz + 7) // 8
    off = o * rb
    bits = np.unpackbits(np.frombuffer(blob[off: off + rbytes], dtype=np.uint8),
                         bitorder="little")[:nnz].astype(bool)
    off += rbytes
    lo = np.frombuffer(blob[off: off + 2 * o], dtype=np.float16)
    hi = np.frombuffer(blob[off + 2 * o: off + 4 * o], dtype=np.float16)
    assert off + 4 * o == len(blob), (off + 4 * o, len(blob))
    is_hi = np.zeros((o, i), dtype=bool)
    is_hi[nzmask] = bits
    mag = np.where(is_hi, hi[:, None], lo[:, None])
    sign = codes.astype(np.int8) - 1
    if raw is not None:
        raw.update(sign=sign, is_hi=is_hi, lo=lo.copy(), hi=hi.copy())
    return (sign.astype(np.float16) * mag).astype(np.float16)


def _intn(blob: bytes, shape, bits: int, raw: dict | None = None) -> np.ndarray:
    o = shape[0]
    total = int(np.prod(shape))
    ncol = total // o
    sc_b = 2 * o
    body, scales = blob[:-sc_b], np.frombuffer(blob[-sc_b:], dtype=np.float16)
    if bits == 8:
        q = np.frombuffer(body, dtype=np.int8).astype(np.int32)[:total]
    elif bits == 6:
        b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        u = np.stack([(packed >> s) & 0x3F for s in (0, 6, 12, 18)], axis=1).ravel()
        q = u[:total].astype(np.int32) - 32
    else:
        raise ValueError(bits)
    if raw is not None:
        raw.update(q=q.reshape(o, ncol).astype(np.int8), scale=scales.copy(), bits=bits)
    w = q.reshape(o, ncol).astype(np.float32) * scales.astype(np.float32)[:, None]
    return w.reshape(shape)


def read_container(path: str, *, with_raw: bool = False):
    """-> (tensors, index) or, with_raw=True, (tensors, index, raw).

    tensors: dict[str, np.ndarray] HF-named; five-value keys are emitted as
    `<module>.weight` like the dense state_dict export.
    raw (with_raw): per record name -> five_value {sign int8 [O,I] in {-1,0,1},
    is_hi bool [O,I], lo fp16 [O], hi fp16 [O]} | intN {q int8 [O,cols],
    scale fp16 [O], bits}.  fp16 records are not in raw."""
    raw_all = {} if with_raw else None
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
        assert header["format"] == FORMAT, header["format"]
        out = {}
        for e in header["index"]:
            blob = fh.read(e["b"])
            assert len(blob) == e["b"]
            k, shape = e["k"], tuple(e["shape"])
            r = {} if with_raw else None
            if k == "five_value":
                out[e["n"] + ".weight"] = _five_value(blob, shape, r)
            elif k.startswith("int"):
                out[e["n"]] = _intn(blob, shape, int(k[3:]), r)
            elif k == "fp16":
                out[e["n"]] = np.frombuffer(blob, dtype=np.float16).reshape(shape).copy()
            else:
                raise ValueError(k)
            if with_raw and r:
                raw_all[e["n"]] = r
        assert fh.read(1) == b"", "trailing bytes"
    if with_raw:
        return out, header["index"], raw_all
    return out, header["index"]
# --- end container reader ----------------------------------------------------------------------------------------------

wer_metric = evaluate.load("wer")

BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"      # config / processor / generation_config; weights come from the container


def container_state_dict(container: str):
    """HF-named fp32 torch state dict (mirrors reference_transformers.py in the model repo)."""
    import re
    tensors, index = read_container(container, with_raw=False)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
        else:
            arr = np.ascontiguousarray(v.astype(np.float32))
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
                arr = arr[:, :, None]                       # kernel-1 pointwise convs stored squeezed
            sd[k] = torch.from_numpy(arr)
    return sd, index


def resolve_container(model_id: str, revision: str | None) -> str:
    """Local dir / local .fermion / Hub repo (unpacks the bps archive if present)."""
    p = Path(model_id)
    if p.is_file():
        return str(p)
    if p.is_dir():
        c = p / "model.fermion"
        assert c.exists(), f"no model.fermion in {p}"
        return str(c)
    local = Path(snapshot_download(model_id, revision=revision))
    if (local / "model.fermion").exists():
        return str(local / "model.fermion")
    arch = next(local.glob("*.bps.tar.zst"), None)
    assert arch is not None, f"neither model.fermion nor a .bps.tar.zst archive in {model_id}"
    import subprocess, tarfile, io
    dest = local / "unpacked"
    if not (dest / "model.fermion").exists():
        dest.mkdir(exist_ok=True)
        # resolve: Hub cache snapshots are symlinks, which zstd silently skips
        proc = subprocess.Popen(["zstd", "-q", "-d", "-c", str(arch.resolve())], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for m in tar:
                if m.isfile():
                    data = tar.extractfile(m).read()
                    out = dest / Path(m.name).name
                    out.write_bytes(data)
        proc.wait()
    return str(dest / "model.fermion")


def main(args):
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch_dtype = getattr(torch, args.dtype)

    from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    container = resolve_container(args.model_id, args.revision)
    cfg = ParakeetTDTConfig.from_pretrained(BASE_MODEL)
    model = ParakeetForTDT(cfg)
    sd, index = container_state_dict(container)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"load: missing {list(missing)[:5]} unexpected {list(unexpected)[:5]}")
    model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
    model.eval().to(torch_dtype).to(args.device)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    print(f"container: {container} ({len(index)} records; five-value encoder + int6 tables, expanded to {args.dtype} for this reference path)")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    sampling_rate = processor.feature_extractor.sampling_rate

    gen_kwargs = {}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    def benchmark(batch, min_new_tokens=None):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(a) / sampling_rate for a in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)
        start_time = time.time()
        inputs = processor(audios, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: (v.to(args.device, torch_dtype) if v.dtype.is_floating_point else v.to(args.device)) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        pred_text = processor.batch_decode(getattr(out, "sequences", out), skip_special_tokens=True)
        runtime = time.time() - start_time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text            # raw; normalisation at scoring time
        batch["references"] = batch["original_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.prepare_data(data_utils.load_data(args), sampling_rate=sampling_rate)
        n = args.warmup_steps * args.batch_size
        warm = dataset.take(n) if args.streaming else dataset.select(range(min(n, len(dataset))))
        for _ in tqdm(iter(warm.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])), desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        dataset = dataset.take(args.max_eval_samples) if args.streaming else dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate)
    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    # Chunked sets (Earnings-22 chunked) carry one reference per parent session: keep parent_id / chunk_index so
    # score_results can reassemble the chunks (this repository's transformers/run_eval.py does the same).
    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    all_results = {"audio_length_s": [], "transcription_time_s": [], "predictions": [], "references": [], "audio_filepath": []}
    if is_chunked:
        all_results.update({key: [] for key in data_utils.CHUNK_METADATA_KEYS})
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"], args.model_id, args.dataset_path, args.dataset or "", args.split,
        audio_length=all_results["audio_length_s"], transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
        extra_fields={key: all_results[key] for key in data_utils.CHUNK_METADATA_KEYS} if is_chunked else None)
    print("Results saved at path:", os.path.abspath(manifest_path))
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    if is_chunked:
        print("chunked dataset: per-chunk WER is not meaningful; WER comes from score_results after chunk merge.", "RTFx:", rtfx)
    else:
        norm_refs = [data_utils.normalizer(r) for r in all_results["references"]]
        norm_preds = [data_utils.normalizer(p) for p in all_results["predictions"]]
        wer = round(100 * wer_metric.compute(references=norm_refs, predictions=norm_preds), 2)
        print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Hub repo id, local model dir, or a local .fermion container")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", type=str, default="", help="dataset config; empty for single-config repos such as VoiceArena/Monsoon_en_IN_test")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=5)
    args = parser.parse_args()
    args.device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    if not args.dataset:            # single-config repos (VoiceArena/Monsoon_en_IN_test): no config name, as this repo's fast-gpu-asr runner does
        args.dataset = None
    main(args)
