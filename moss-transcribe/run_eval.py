"""
Open ASR Leaderboard entry: MOSS-Transcribe-preview-2B.

Pipeline:
  - librosa.load @ 16 kHz
  - WhisperFeatureExtractor log-mel (n_fft=400, hop=160, dim=128)
  - MossProcessor with a single default chat template (identical across all
    datasets, no per-dataset style switching)
  - MossForCausalLM.generate(greedy, num_beams=1, eos=processor.end_token_id)
"""
import argparse
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

import evaluate
from normalizer import data_utils

wer_metric = evaluate.load("wer")

SAMPLE_RATE = 16000
MEL_DIM = 128
N_FFT = 400
HOP = 160


def build_processor(model_id, tokenizer, revision=None):
    """Reproduce WhisperLogMelMossProcessor.__call__ behavior."""
    MossProcessor = get_class_from_dynamic_module(
        "processing_Moss.MossProcessor", model_id, revision=revision
    )
    MelConfig = get_class_from_dynamic_module(
        "processing_Moss.MelConfig", model_id, revision=revision
    )
    mel_cfg = MelConfig(mel_sr=SAMPLE_RATE, mel_dim=MEL_DIM, mel_n_fft=N_FFT, mel_hop_length=HOP)
    processor = MossProcessor(tokenizer, config=mel_cfg, enable_time_marker=False)
    template_path = os.path.join(os.path.dirname(__file__), "chat_template_default.py")
    processor.load_template(template_path)

    fe = WhisperFeatureExtractor(
        feature_size=MEL_DIM, sampling_rate=SAMPLE_RATE, hop_length=HOP, n_fft=N_FFT
    )
    return processor, fe


def encode_one(audio_np: np.ndarray, processor, fe):
    """Single-sample encode: returns (input_ids[L], audio_mask[L], mel[128,T], T)."""
    wav = audio_np.astype(np.float32)
    try:
        mel = fe._np_extract_fbank_features(wav[None, ...], device="cpu")[0]
    except TypeError:
        mel = fe._np_extract_fbank_features(wav[None, ...])[0]
    mel = torch.from_numpy(mel).to(processor.config.mel_dtype)
    if mel.dim() == 3:
        mel = mel.squeeze(0)

    T = mel.shape[-1]
    num_audio_tokens = processor._get_feat_extract_output_lengths(T)
    if processor.chat_template is not None:
        ids, mask = processor._build_input_from_template(num_audio_tokens)
    else:
        ids, mask = processor._build_input_legacy(num_audio_tokens)
    return ids, mask, mel, T


def encode_batch(audios, processor, fe, device, model_dtype) -> BatchEncoding:
    """Concat mel along time dim, left-pad input_ids per sample for batched generate."""
    encs = [encode_one(a, processor, fe) for a in audios]
    L = max(len(e[0]) for e in encs)
    bs = len(encs)
    pad_id = processor.tokenizer.pad_token_id or processor.end_token_id

    input_ids = torch.full((bs, L), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((bs, L), dtype=torch.long)
    audio_mask = torch.zeros((bs, L), dtype=torch.bool)
    seq_lens = torch.zeros((bs,), dtype=torch.long)
    mel_parts = []
    for i, (ids, m, mel, T) in enumerate(encs):
        li = len(ids)
        input_ids[i, -li:]  = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, -li:]  = 1
        audio_mask[i, -li:] = torch.tensor(m, dtype=torch.bool)
        seq_lens[i] = T
        mel_parts.append(mel)
    audio_data = torch.cat(mel_parts, dim=-1)

    return BatchEncoding(data={
        "input_ids": input_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "audio_data": audio_data.to(device).to(model_dtype),
        "audio_data_seqlens": seq_lens.to(device),
        "audio_input_mask": audio_mask.to(device),
    })


def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype="auto", trust_remote_code=True, revision=args.model_revision
    ).to(device)
    model.eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=args.model_revision
    )
    processor, fe = build_processor(args.model_id, tokenizer, revision=args.model_revision)
    eos_ids = [processor.end_token_id]

    def benchmark(batch):
        # Open ASR Leaderboard hands us audio dicts at 16 kHz already; resample defensively.
        sr_in = batch["audio"][0]["sampling_rate"]
        audios = []
        for a in batch["audio"]:
            wav = a["array"]
            if sr_in != SAMPLE_RATE:
                wav = librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=SAMPLE_RATE)
            audios.append(wav)
        batch["audio_length_s"] = [len(w) / SAMPLE_RATE for w in audios]
        mb = len(audios)
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, mb)

        torch.cuda.synchronize(device=device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        inputs = encode_batch(audios, processor, fe, device, model.dtype)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=eos_ids,
            )
        new_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
        preds = [t.strip() for t in processor.batch_decode(new_ids, skip_special_tokens=True)]

        end_event.record()
        torch.cuda.synchronize(device=device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        batch["transcription_time_s"] = mb * [runtime / mb]
        # Save raw (unnormalized) outputs; normalization is applied at scoring time.
        batch["predictions"] = preds
        batch["references"] = batch["original_text"]
        return batch

    # ---- Warm-up ----
    if args.warmup_steps:
        warmup = data_utils.load_data(args)
        warmup = data_utils.prepare_data(warmup)
        n = args.warmup_steps * args.batch_size
        warmup = warmup.take(n) if args.streaming else warmup.select(range(min(n, len(warmup))))
        for _ in tqdm(iter(warmup.map(benchmark, batch_size=args.batch_size, batched=True)), desc="Warmup"):
            pass

    # ---- Eval ----
    ds = data_utils.load_data(args)
    ds = data_utils.prepare_data(ds)
    if args.max_eval_samples and args.max_eval_samples > 0:
        ds = ds.take(args.max_eval_samples) if args.streaming else ds.select(
            range(min(args.max_eval_samples, len(ds)))
        )
    ds = ds.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)

    res = {"audio_length_s": [], "transcription_time_s": [], "predictions": [], "references": [], "audio_filepath": []}
    if is_chunked:
        res.update({k: [] for k in data_utils.CHUNK_METADATA_KEYS})
    for r in tqdm(iter(ds), desc="Samples"):
        for k in res:
            res[k].append(r[k])

    manifest = data_utils.write_manifest(
        res["references"], res["predictions"], args.model_id,
        args.dataset_path, args.dataset, args.split,
        audio_length=res["audio_length_s"], transcription_time=res["transcription_time_s"],
        audio_filepaths=res["audio_filepath"],
        extra_fields={k: res[k] for k in data_utils.CHUNK_METADATA_KEYS} if is_chunked else None,
    )
    print("Manifest:", os.path.abspath(manifest))

    # Chunked datasets are scored per parent session, not per chunk.
    if is_chunked:
        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest))
        references = [s["text"] for s in sessions]
        predictions = [s["pred_text"] for s in sessions]
    else:
        references = res["references"]
        predictions = res["predictions"]

    # Normalize raw references/predictions at scoring time.
    norm_refs = [data_utils.normalizer(r) for r in references]
    norm_preds = [data_utils.normalizer(p) for p in predictions]
    wer = round(100 * wer_metric.compute(references=norm_refs, predictions=norm_preds), 2)
    rtfx = round(sum(res["audio_length_s"]) / sum(res["transcription_time_s"]), 2)
    print(f"WER: {wer}%  RTFx: {rtfx}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--model_revision", type=str, default="c4b3988677df13c14e79d9db59f356ed761db366",
                   help="Pin the model repo revision (commit) for trust_remote_code stability.")
    p.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--streaming", action="store_true", help="Stream the dataset instead of downloading it.")
    p.add_argument("--warmup_steps", type=int, default=10)
    args = p.parse_args()
    main(args)