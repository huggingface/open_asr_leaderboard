"""Open ASR Leaderboard evaluator for MOSS-Transcribe-Diarize.

This keeps the Musci-ASR evaluation shell (HF Dataset, batched map, standard
Open ASR manifest) and replaces only the model-specific inference path with
the official MOSS processor/generation interface.
"""

from __future__ import annotations

import argparse
import copy
import re
import time
from typing import Any

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.logits_process import LogitsProcessor

from normalizer import data_utils


SAMPLE_RATE = 16_000
MIN_GENERATED_TOKENS = 128
TOKENS_PER_AUDIO_SECOND = 12
TOKEN_BUDGET_MARGIN = 64
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def plain_transcript(text: str) -> str:
    """Remove MOSS timestamp/speaker markup before Open ASR WER scoring."""
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|?[^>|\n]*\|?>", " ", text)
    text = re.sub(r"\[S0*\d+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:-->|->|-)\s*\d+(?:\.\d+)?\b", " ", text)
    text = re.sub(
        r"\[(?:\d{1,2}:)?\d{1,2}(?:\.\d+)?\s*[-,]\s*"
        r"(?:\d{1,2}:)?\d{1,2}(?:\.\d+)?\]",
        " ",
        text,
    )
    text = re.sub(r"\[\s*\d+(?:\.\d+)?\s*\]", " ", text)
    text = re.sub(
        r"\(\s*(?:\d{1,2}:)?\d{1,2}(?:\.\d+)?\s*[-,]\s*"
        r"(?:\d{1,2}:)?\d{1,2}(?:\.\d+)?\s*\)",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def build_messages() -> list[dict[str, Any]]:
    # The chat template only needs the audio item type. The waveform itself
    # is supplied through processor(..., audio=[waveform]) below.
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": True},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]


def _decode_audio_item(item: Any) -> tuple[np.ndarray, int]:
    """Handle the dict and AudioDecoder forms returned by datasets.Audio."""
    if isinstance(item, dict):
        if item.get("array") is not None:
            return np.asarray(item["array"], dtype=np.float32), int(
                item.get("sampling_rate", SAMPLE_RATE)
            )
        if item.get("path"):
            waveform, sr = librosa.load(item["path"], sr=None, mono=True)
            return waveform.astype(np.float32), int(sr)

    if hasattr(item, "get_all_samples"):
        samples = item.get_all_samples()
        waveform = samples.data
        if torch.is_tensor(waveform):
            waveform = waveform.detach().cpu().numpy()
        return np.asarray(waveform, dtype=np.float32).squeeze(), int(samples.sample_rate)

    if isinstance(item, str):
        waveform, sr = librosa.load(item, sr=None, mono=True)
        return waveform.astype(np.float32), int(sr)

    raise TypeError(f"Unsupported audio item type: {type(item)!r}")


def _to_16khz(item: Any) -> np.ndarray:
    waveform, sampling_rate = _decode_audio_item(item)
    waveform = np.asarray(waveform, dtype=np.float32).squeeze()
    if waveform.ndim == 2:
        # Multi-channel audio (e.g. some Earnings22 clips): downmix to mono.
        # Channels are the smaller axis; average across it.
        channel_axis = int(np.argmin(waveform.shape))
        waveform = waveform.mean(axis=channel_axis)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("Audio is empty")
    if sampling_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sampling_rate, target_sr=SAMPLE_RATE
        )
    return np.asarray(waveform, dtype=np.float32)


def _move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> Any:
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def _left_pad_inputs(inputs: Any, pad_token_id: int) -> None:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    padded_ids = torch.full_like(input_ids, pad_token_id)
    padded_mask = torch.zeros_like(attention_mask)
    for index, length in enumerate(attention_mask.sum(dim=1).tolist()):
        length = int(length)
        padded_ids[index, -length:] = input_ids[index, :length]
        padded_mask[index, -length:] = 1
    inputs["input_ids"] = padded_ids
    inputs["attention_mask"] = padded_mask


def load_model(args: argparse.Namespace):
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
        dtype="auto",
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0
    print(
        f"[moss-transcribe-diarize] model={args.model_id} "
        f"revision={args.model_revision} device={device} dtype={dtype}",
        flush=True,
    )
    print(
        f"Model size: {sum(parameter.numel() for parameter in model.parameters()) / 1e9:.2f}B parameters",
        flush=True,
    )
    return model, processor, device, dtype, int(pad_token_id)


def unfinished_generation_indices(
    generated_ids: torch.Tensor,
    eos_token_id,
    generation_cap: int,
) -> list[int]:
    if int(generated_ids.shape[1]) < generation_cap:
        return []
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = {int(token_id) for token_id in eos_token_id}
    return [
        index
        for index, token_ids in enumerate(generated_ids.tolist())
        if not any(token_id in eos_token_ids for token_id in token_ids)
    ]


class PerSampleTokenBudgetLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        prompt_width: int,
        token_budgets: list[int],
        eos_token_id: int,
        eos_token_ids: set[int],
    ):
        self.prompt_width = prompt_width
        self.token_budgets = token_budgets
        self.eos_token_id = eos_token_id
        self.eos_token_ids = eos_token_ids
        self.forced_indices: set[int] = set()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_tokens = int(input_ids.shape[1]) - self.prompt_width
        for index, token_budget in enumerate(self.token_budgets):
            if generated_tokens < token_budget:
                continue
            if any(
                int(token_id) in self.eos_token_ids
                for token_id in input_ids[index, self.prompt_width :]
            ):
                continue
            scores[index].fill_(torch.finfo(scores.dtype).min)
            scores[index, self.eos_token_id] = 0
            self.forced_indices.add(index)
        return scores


def token_budget_for_duration(
    duration: float,
    *,
    max_new_tokens: int,
    batch_max_new_tokens: int,
) -> int:
    budget = max(
        MIN_GENERATED_TOKENS,
        int(np.ceil(duration * TOKENS_PER_AUDIO_SECOND)) + TOKEN_BUDGET_MARGIN,
    )
    return min(max_new_tokens, batch_max_new_tokens, budget)


def transcribe_batch(
    waveforms: list[np.ndarray],
    durations: list[float],
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    pad_token_id: int,
    max_new_tokens: int,
    batch_max_new_tokens: int,
) -> tuple[list[str], float, dict[str, Any]]:
    messages = build_messages()
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(
        device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        audio_kwargs = {"device": str(device)} if device.type == "cuda" else {}
        inputs = processor(
            text=[prompt] * len(waveforms),
            audio=waveforms,
            max_length=131072,
            audio_kwargs=audio_kwargs,
            return_tensors="pt",
        )
        _left_pad_inputs(inputs, pad_token_id)
        inputs = _move_inputs(inputs, device, dtype)
        prompt_length = inputs["input_ids"].shape[1]

        generation_config = copy.deepcopy(model.generation_config)
        generation_cap = (
            max_new_tokens
            if len(waveforms) == 1
            else min(max_new_tokens, batch_max_new_tokens)
        )
        generation_config.max_new_tokens = generation_cap
        generation_config.do_sample = False
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = processor.tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = {int(token_id) for token_id in eos_token_id}
            force_eos_token_id = int(eos_token_id[0])
        else:
            eos_token_ids = {int(eos_token_id)}
            force_eos_token_id = int(eos_token_id)
        token_budgets = [
            token_budget_for_duration(
                duration,
                max_new_tokens=max_new_tokens,
                batch_max_new_tokens=batch_max_new_tokens,
            )
            for duration in durations
        ]
        token_budget_processor = PerSampleTokenBudgetLogitsProcessor(
            prompt_width=prompt_length,
            token_budgets=token_budgets,
            eos_token_id=force_eos_token_id,
            eos_token_ids=eos_token_ids,
        )
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
            generation_config=generation_config,
            logits_processor=[token_budget_processor],
        )
        generated_ids = output_ids[:, prompt_length:]
        decoded = processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    predictions = [plain_transcript(text) for text in decoded]
    unfinished = unfinished_generation_indices(
        generated_ids,
        generation_config.eos_token_id,
        generation_cap,
    )
    fallback_elapsed = 0.0
    unfinished_after_fallback = 0
    if unfinished and generation_cap < max_new_tokens:
        print(
            f"Batch cap reached by {len(unfinished)}/{len(waveforms)} samples; "
            "retrying those samples individually",
            flush=True,
        )
        for index in unfinished:
            retry_predictions, retry_elapsed, retry_stats = transcribe_batch(
                [waveforms[index]],
                [durations[index]],
                model,
                processor,
                device,
                dtype,
                pad_token_id,
                max_new_tokens,
                batch_max_new_tokens,
            )
            predictions[index] = retry_predictions[0]
            fallback_elapsed += retry_elapsed
            unfinished_after_fallback += retry_stats[
                "unfinished_after_fallback"
            ]
    else:
        unfinished_after_fallback = len(unfinished)
    elapsed += fallback_elapsed
    return predictions, elapsed, {
        "generation_cap": generation_cap,
        "min_token_budget": min(token_budgets),
        "max_token_budget": max(token_budgets),
        "forced_eos_samples": len(token_budget_processor.forced_indices),
        "fallback_samples": len(unfinished) if generation_cap < max_new_tokens else 0,
        "fallback_inference_sec": fallback_elapsed,
        "unfinished_after_fallback": unfinished_after_fallback,
    }


def main(args: argparse.Namespace) -> int:
    model, processor, device, dtype, pad_token_id = load_model(args)

    def benchmark(batch):
        waveforms = [_to_16khz(item) for item in batch["audio"]]
        audio_lengths = [len(waveform) / SAMPLE_RATE for waveform in waveforms]
        minibatch_size = len(waveforms)
        try:
            predictions, elapsed, inference_stats = transcribe_batch(
                waveforms,
                audio_lengths,
                model,
                processor,
                device,
                dtype,
                pad_token_id,
                args.max_new_tokens,
                args.batch_max_new_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"MOSS batched inference failed for {minibatch_size} items: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if len(predictions) != minibatch_size:
            raise RuntimeError(
                f"Expected {minibatch_size} predictions, got {len(predictions)}"
            )

        batch["audio_length_s"] = audio_lengths
        batch["transcription_time_s"] = [elapsed / minibatch_size] * minibatch_size
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(
            batch,
            minibatch_size,
        )
        # Keep original references and raw-cleaned predictions here. The
        # standard Space normalizer performs the official scoring normalization.
        batch["references"] = batch["original_text"]
        batch["predictions"] = predictions
        print(
            f"Batch size: {minibatch_size}, time: {elapsed:.3f}s, "
            f"audio: {sum(audio_lengths):.3f}s, "
            f"RTFx: {sum(audio_lengths) / elapsed:.2f}, "
            f"forced EOS: {inference_stats['forced_eos_samples']}, "
            f"fallbacks: {inference_stats['fallback_samples']}",
            flush=True,
        )
        return batch

    def load_prepared_dataset():
        dataset = data_utils.load_data(args)
        return data_utils.prepare_data(dataset)

    if args.warmup_steps:
        warmup = load_prepared_dataset()
        count = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup = warmup.take(count)
        else:
            warmup = warmup.select(range(min(count, len(warmup))))
        for _ in tqdm(
            iter(warmup.map(benchmark, batch_size=args.batch_size, batched=True)),
            desc="Warmup",
        ):
            pass

    dataset = load_prepared_dataset()
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(
                range(min(args.max_eval_samples, len(dataset)))
            )

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "audio_filepath": [],
        "predictions": [],
        "references": [],
    }
    for result in tqdm(iter(dataset), desc="Samples"):
        for key in results:
            results[key].append(result[key])

    if not results["references"]:
        raise RuntimeError("Evaluation produced zero samples")
    if any(pred is None for pred in results["predictions"]):
        raise RuntimeError("Evaluation produced a null prediction")

    manifest = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
        audio_filepaths=results["audio_filepath"],
    )
    print("Manifest:", manifest, flush=True)

    total_duration = sum(results["audio_length_s"])
    total_time = sum(results["transcription_time_s"])
    rtfx = total_duration / total_time if total_time else 0.0
    print(
        f"Samples: {len(results['references'])}  RTFx: {rtfx:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--model_revision",
        default="e5118b411bf5a77d7a90c4941066bec93c967312",
    )
    parser.add_argument(
        "--dataset_path",
        default="hf-audio/open-asr-leaderboard",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="True model inference batch size.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_max_new_tokens", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset instead of downloading it.",
    )
    parser.add_argument("--warmup_steps", type=int, default=1)
    args = parser.parse_args()
    if args.batch_max_new_tokens <= 0:
        parser.error("--batch_max_new_tokens must be positive")
    if args.batch_max_new_tokens > args.max_new_tokens:
        parser.error("--batch_max_new_tokens cannot exceed --max_new_tokens")
    raise SystemExit(main(args))
