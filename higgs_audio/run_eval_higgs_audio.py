"""Evaluation script for Higgs Audio v3 models on ESB benchmark datasets.

Usage:
    python run_eval_higgs_audio.py \
        --model_id bosonai/higgs-audio-v3-8b-stt \
        --dataset_path hf-audio/esb-datasets-test-only-sorted \
        --dataset ami --split test --device 0
"""

import argparse
import json
import os
import time

import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperProcessor
from normalizer import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    # Load model (custom architecture requires trust_remote_code)
    print(f"Loading model {args.model_id}...")
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.eval()

    # Audio preprocessing
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.data_types import AudioContent, Message
    from boson_multimodal.dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample_qwen
    from functools import partial

    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_proc,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=1,
        audio_num_codebooks=config.audio_num_codebooks,
        chunk_size_seconds=getattr(config, "chunk_size_seconds", 30),
        encoder_padding_method=getattr(config, "encoder_padding_method", "max_length"),
    )

    user_prompt = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."

    # Load dataset
    print(f"Loading dataset {args.dataset_path}/{args.dataset} ({args.split})...")
    dataset = load_dataset(args.dataset_path, args.dataset, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_eval_samples > 0:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    print(f"Evaluating {len(dataset)} samples...")

    predictions = []
    references = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    for i, sample in enumerate(dataset):
        audio = sample["audio"]
        audio_np = np.array(audio["array"], dtype=np.float32)
        ref_text = sample.get("norm_text", sample.get("text", ""))
        total_audio_duration += len(audio_np) / audio["sampling_rate"]

        # Build input
        messages = [Message(role="user", content=[user_prompt, AudioContent(audio_url="placeholder")])]
        chatml = ChatMLSample(messages=messages)
        prep_fn = partial(prepare_chatml_sample_qwen, enable_thinking=True)
        input_tokens, _, _, _ = prep_fn(chatml, tokenizer, add_generation_prompt=True)

        ds_sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=torch.LongTensor([-100] * len(input_tokens)),
            audio_ids_concat=None, audio_ids_start=None,
            audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
            audio_waveforms_start=torch.tensor([0]),
            audio_sample_rate=torch.tensor([16000]),
            audio_speaker_indices=torch.tensor([0]),
        )
        from dataclasses import asdict
        batch = asdict(collator([ds_sample]))
        batch = {k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Generate
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch.get("audio_features"),
                audio_feature_attention_mask=batch.get("audio_feature_attention_mask"),
                max_new_tokens=1024,
                do_sample=False,
            )
        total_inference_time += time.time() - t0

        # Decode — strip thinking tokens
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        if "</think>" in full_text:
            full_text = full_text.split("</think>")[-1]
        pred = full_text.replace("<|im_end|>", "").strip()

        # Normalize
        pred_norm = normalizer(pred)
        ref_norm = normalizer(ref_text)

        predictions.append(pred_norm)
        references.append(ref_norm)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dataset)} done", flush=True)

    # Compute WER
    from jiwer import wer
    wer_score = round(wer(references, predictions) * 100, 2)
    rtfx = round(total_audio_duration / total_inference_time, 2) if total_inference_time > 0 else 0

    print(f"\nResults for {args.model_id} on {args.dataset}:")
    print(f"  WER: {wer_score}%")
    print(f"  RTFx: {rtfx}")

    # Save results
    manifest = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "split": args.split,
        "wer": wer_score,
        "rtfx": rtfx,
        "num_samples": len(dataset),
        "predictions": predictions,
        "references": references,
    }

    out_dir = os.path.join("results", args.model_id.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
