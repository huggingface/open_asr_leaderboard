import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import evaluate
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from normalizer import data_utils
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

wer_metric = evaluate.load("wer")

SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)"
    r")\|>"
)
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")
CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")


class BlockTokenIdsFromLogitsProcessor(LogitsProcessor):
    def __init__(self, block_from_id: int | None, block_token_ids: Iterable[int] | None = None):
        self.block_from_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
        self.block_token_ids = sorted(set(int(token_id) for token_id in (block_token_ids or [])))

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        vocab_size = scores.shape[-1]
        if self.block_from_id is not None and self.block_from_id < vocab_size:
            scores[:, self.block_from_id :] = -float("inf")
        valid_token_ids = [token_id for token_id in self.block_token_ids if 0 <= token_id < vocab_size]
        if valid_token_ids:
            scores[:, valid_token_ids] = -float("inf")
        return scores


def normalize_token_ids(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        return [int(token_id) for token_id in token_ids if token_id is not None]
    return [int(token_ids)]


def build_eos_token_ids(tokenizer: Any) -> list[int]:
    eos_ids = []
    eos_ids.extend(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    for marker in TURN_END_MARKERS:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.append(int(token_id))
    return list(dict.fromkeys(eos_ids))


def build_asr_keep_token_ids(model: Any, tokenizer: Any) -> list[int]:
    keep_token_ids = set()
    keep_token_ids.update(normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    keep_token_ids.update(normalize_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None)))
    keep_token_ids.update(normalize_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None)))
    return sorted(keep_token_ids)


def build_asr_extra_block_token_ids(
    tokenizer: Any,
    keep_token_ids: Iterable[int] | None = None,
    block_from_id: int | None = None,
) -> list[int]:
    keep = set(int(token_id) for token_id in (keep_token_ids or []))
    max_control_token_id = None if block_from_id is None or int(block_from_id) < 0 else int(block_from_id)
    block_token_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None)
    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for token_id, token_meta in added_tokens_decoder.items():
        token_id = int(token_id)
        if max_control_token_id is not None and token_id >= max_control_token_id:
            continue
        token_content = getattr(token_meta, "content", None)
        if token_content is None and isinstance(token_meta, dict):
            token_content = token_meta.get("content")
        if token_content and CONTROL_TOKEN_PATTERN.match(token_content):
            block_token_ids.add(token_id)
    block_token_ids.difference_update(keep)
    return sorted(block_token_ids)


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        return {key: value[key] for key in value.keys()}
    raise TypeError(f"Unexpected processor output type: {type(value)}")


def truncate_generation_text(text: str) -> str:
    if not text:
        return ""
    cut = len(text)
    for marker in TURN_END_MARKERS:
        index = text.find(marker)
        if index != -1 and index < cut:
            cut = index
    return text[:cut].strip()


def remove_special_tokens(text: str) -> str:
    if not text:
        return ""
    if "<|text|>" in text:
        text = text.split("<|text|>", 1)[1]
    return SPECIAL_TOKEN_PATTERN.sub("", text).strip()


def normalize_prediction_text(text: str) -> str:
    if not text:
        return ""
    text = truncate_generation_text(text)
    text = remove_special_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def resolve_device(device_index: int) -> str:
    if device_index >= 0:
        return f"cuda:{device_index}"
    return "cpu"


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    if not device.startswith("cuda") and mapping[dtype_name] != torch.float32:
        print(f"Warning: dtype={dtype_name} on {device} is not well supported. Falling back to float32.")
        return torch.float32
    return mapping[dtype_name]


def load_model(model_id: str, device: str, torch_dtype: torch.dtype, attn_impl: str):
    if attn_impl == "auto":
        candidates = ["flash_attention_2", "sdpa"] if device.startswith("cuda") else ["eager"]
    else:
        candidates = [attn_impl]
        if attn_impl == "flash_attention_2":
            candidates.extend(["sdpa", "eager"] if device.startswith("cuda") else ["eager"])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation=candidate,
            ).to(device)
            model.eval()
            return model, candidate
        except (ImportError, RuntimeError, ValueError) as exc:
            message = str(exc)
            can_fallback = candidate == "flash_attention_2" and (
                "flash_attn" in message or "FlashAttention" in message
            )
            if not can_fallback:
                raise
            print(f"Warning: attn_impl={candidate} unavailable ({message.splitlines()[0]}). Falling back.")
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to load model with any attention implementation.")


def apply_audio_gain(audios: Tensor, gain: float) -> Tensor:
    gain = float(gain)
    if gain == 1.0:
        return audios
    return torch.clamp(audios.float() * gain, min=-1.0, max=1.0)


def prepare_dataset(args):
    if args.dataset_revision:
        dataset = load_dataset(
            args.dataset_path,
            args.dataset,
            split=args.split,
            streaming=args.streaming,
            token=True,
            revision=args.dataset_revision,
        )
    else:
        dataset = data_utils.load_data(args)

    if args.audio_decode == "datasets":
        return data_utils.prepare_data(dataset, sampling_rate=args.target_sr)

    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.map(data_utils.normalize)
    return dataset.filter(data_utils.is_target_text_in_range, input_columns=["norm_text"])


def iter_local_parquet_batches(
    local_parquet_dir: str,
    *,
    batch_size: int,
    skip_samples: int,
    max_samples: int | None,
) -> Iterable[dict[str, list[Any]]]:
    parquet_files = sorted(Path(local_parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {local_parquet_dir}")

    yielded = 0
    seen = 0
    batch = {"audio": [], "norm_text": []}
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for row_group_index in range(pf.num_row_groups):
            rows = pf.read_row_group(row_group_index).to_pylist()
            for row in rows:
                normalized_text = data_utils.normalizer(data_utils.get_text(row))
                if not data_utils.is_target_text_in_range(normalized_text):
                    continue
                if seen < skip_samples:
                    seen += 1
                    continue
                if max_samples is not None and max_samples > 0 and yielded >= max_samples:
                    if batch["audio"]:
                        yield batch
                    return

                batch["audio"].append(row["audio"])
                batch["norm_text"].append(normalized_text)
                yielded += 1
                seen += 1
                if len(batch["audio"]) >= batch_size:
                    yield batch
                    batch = {"audio": [], "norm_text": []}

    if batch["audio"]:
        yield batch


def build_manifest_path(model_id: str, dataset_path: str, dataset_name: str, split: str, suffix: str | None = None) -> str:
    model_id = model_id.replace("/", "-")
    dataset_path = dataset_path.replace("/", "-")
    dataset_name = dataset_name.replace("/", "-")
    split = split.replace("/", "-")

    basedir = "./results/"
    os.makedirs(basedir, exist_ok=True)
    stem = f"MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}"
    if suffix:
        suffix = suffix.strip().replace("/", "-")
        stem = f"{stem}_{suffix}"
    return os.path.join(basedir, f"{stem}.jsonl")


def write_manifest_records(
    manifest_path: str,
    references: list[str],
    predictions: list[str],
    audio_length: list[float],
    transcription_time: list[float],
    append: bool = False,
) -> None:
    if len(references) != len(predictions):
        raise ValueError(
            f"The number of samples in `references` ({len(references)}) "
            f"must match `predictions` ({len(predictions)})."
        )
    if len(audio_length) != len(references):
        raise ValueError(
            f"The number of samples in `audio_length` ({len(audio_length)}) "
            f"must match `references` ({len(references)})."
        )
    if len(transcription_time) != len(references):
        raise ValueError(
            f"The number of samples in `transcription_time` ({len(transcription_time)}) "
            f"must match `references` ({len(references)})."
        )

    mode = "a" if append else "w"
    with open(manifest_path, mode, encoding="utf-8") as f:
        for idx, (text, transcript, duration, runtime) in enumerate(
            zip(references, predictions, audio_length, transcription_time)
        ):
            datum = {
                "audio_filepath": f"sample_{idx}",
                "duration": duration,
                "time": runtime,
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")


def decode_audio(audio: dict[str, Any], target_sr: int) -> tuple[np.ndarray, int]:
    if "array" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio["sampling_rate"])
    else:
        audio_bytes = audio.get("bytes")
        if audio_bytes is not None:
            array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio_path = audio.get("path")
            if not audio_path:
                raise ValueError(f"Audio sample must contain decoded array, bytes, or path. Got keys: {list(audio)}")
            array, sampling_rate = sf.read(audio_path, dtype="float32")

    if array.ndim > 1:
        array = array.mean(axis=1)
    if sampling_rate != target_sr:
        import librosa

        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr
    return np.asarray(array, dtype=np.float32), sampling_rate


class TemporaryWavBatch:
    def __init__(self, audios: list[np.ndarray], sampling_rates: list[int], enabled: bool) -> None:
        self.audios = audios
        self.sampling_rates = sampling_rates
        self.enabled = enabled
        self.tempdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> list[str | dict[str, Any]]:
        if not self.enabled:
            return [
                {"array": audio, "sampling_rate": sampling_rate}
                for audio, sampling_rate in zip(self.audios, self.sampling_rates)
            ]

        self.tempdir = tempfile.TemporaryDirectory(prefix="ark_asr_eval_")
        audio_paths = []
        for index, (audio, sampling_rate) in enumerate(zip(self.audios, self.sampling_rates)):
            path = Path(self.tempdir.name) / f"sample_{index}.wav"
            sf.write(path, audio, sampling_rate)
            audio_paths.append(str(path))
        return audio_paths

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.tempdir is not None:
            self.tempdir.cleanup()


class ArkAsrInferencer:
    def __init__(
        self,
        model_id: str,
        processor_id: str | None,
        *,
        device: str,
        dtype: str,
        attn_impl: str,
        padding_side: str,
        asr_block_token_id_from: int,
    ) -> None:
        self.device = device
        self.torch_dtype = resolve_torch_dtype(dtype, device)
        self.model, self.resolved_attn_impl = load_model(model_id, device, self.torch_dtype, attn_impl)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = padding_side
        self.processor = AutoProcessor.from_pretrained(
            processor_id or model_id,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if hasattr(self.processor, "tokenizer"):
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.tokenizer.pad_token_id
            self.processor.tokenizer.padding_side = padding_side
        self.eos_token_ids = build_eos_token_ids(self.tokenizer)
        keep_token_ids = build_asr_keep_token_ids(self.model, self.tokenizer)
        self.extra_block_token_ids = build_asr_extra_block_token_ids(
            self.tokenizer,
            keep_token_ids=keep_token_ids,
            block_from_id=asr_block_token_id_from,
        )
        self.asr_block_token_id_from = asr_block_token_id_from

    def transcribe(
        self,
        audio_inputs: list[str | dict[str, Any]],
        *,
        target_sr: int,
        max_audio_seconds: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        audio_gain: float,
    ) -> list[str]:
        conversations = []
        for audio_input in audio_inputs:
            audio_content = {"type": "audio"}
            if isinstance(audio_input, str):
                audio_content["path"] = audio_input
            else:
                audio_content.update(audio_input)
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            audio_content,
                            {"type": "text", "text": "Please transcribe this audio."},
                        ],
                    }
                ]
            )

        inputs_raw = self.processor.apply_chat_template(
            conversations,
            return_tensors="pt",
            sampling_rate=target_sr,
            audio_padding="longest",
            add_generation_prompt=True,
            text_kwargs={"padding": "longest"},
            audio_max_length=int(max_audio_seconds * target_sr),
        )
        if torch.is_tensor(inputs_raw):
            raise RuntimeError("ASR apply_chat_template returned Tensor-only; audio was not encoded.")
        inputs = as_dict(inputs_raw)
        if "audios" not in inputs:
            raise RuntimeError(f"ASR inputs missing 'audios'; processor keys={list(inputs.keys())}")
        if "attention_mask" not in inputs and "input_ids" in inputs and torch.is_tensor(inputs["input_ids"]):
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue
            if key == "audios":
                inputs[key] = apply_audio_gain(value, audio_gain).to(device=self.device, dtype=self.torch_dtype)
            else:
                inputs[key] = value.to(self.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.eos_token_ids:
            generate_kwargs["eos_token_id"] = self.eos_token_ids
        if do_sample:
            generate_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if self.asr_block_token_id_from >= 0 or self.extra_block_token_ids:
            generate_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    BlockTokenIdsFromLogitsProcessor(
                        block_from_id=self.asr_block_token_id_from,
                        block_token_ids=self.extra_block_token_ids,
                    )
                ]
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        input_ids = inputs["input_ids"]
        predictions = []
        for index, output in enumerate(outputs):
            generated_ids = output[len(input_ids[index].tolist()) :]
            prediction_raw = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            predictions.append(normalize_prediction_text(prediction_raw))
        return predictions


def main(args):
    device = resolve_device(args.device)
    inferencer = ArkAsrInferencer(
        args.model_id,
        args.processor_id,
        device=device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        padding_side=args.padding_side,
        asr_block_token_id_from=args.asr_block_token_id_from,
    )
    print(f"Loaded {args.model_id}; device={device}; attn={inferencer.resolved_attn_impl}; dtype={inferencer.torch_dtype}")

    def benchmark(batch):
        decoded = [decode_audio(audio, args.target_sr) for audio in batch["audio"]]
        audios = [audio for audio, _ in decoded]
        sampling_rates = [sampling_rate for _, sampling_rate in decoded]
        batch["audio_length_s"] = [len(audio) / sampling_rate for audio, sampling_rate in zip(audios, sampling_rates)]
        minibatch_size = len(audios)

        with TemporaryWavBatch(audios, sampling_rates, enabled=args.audio_input == "temp_wav") as audio_inputs:
            start_time = time.time()
            pred_text = inferencer.transcribe(
                audio_inputs,
                target_sr=args.target_sr,
                max_audio_seconds=args.max_audio_seconds,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                audio_gain=args.audio_gain,
            )
            runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        if args.local_parquet_dir:
            print("Skipping warmup for local parquet input.")
        else:
            warmup_dataset = prepare_dataset(args)
            num_warmup_samples = args.warmup_steps * args.batch_size
            if args.streaming:
                warmup_dataset = warmup_dataset.take(num_warmup_samples)
            else:
                warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
            warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

            for _ in tqdm(warmup_dataset, desc="Warming up..."):
                continue

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    manifest_path = None
    if args.incremental_manifest:
        manifest_path = build_manifest_path(
            args.model_id,
            args.dataset_path,
            args.dataset,
            args.split,
            suffix=args.manifest_suffix,
        )
        open(manifest_path, "w", encoding="utf-8").close()

    if args.local_parquet_dir:
        result_iter = (
            benchmark(batch)
            for batch in iter_local_parquet_batches(
                args.local_parquet_dir,
                batch_size=args.batch_size,
                skip_samples=args.skip_eval_samples,
                max_samples=args.max_eval_samples,
            )
        )
        for batch_result in tqdm(result_iter, desc="Batches..."):
            batch_len = len(batch_result["references"])
            for index in range(batch_len):
                for key in all_results:
                    all_results[key].append(batch_result[key][index])
                if args.incremental_manifest:
                    write_manifest_records(
                        manifest_path,
                        [batch_result["references"][index]],
                        [batch_result["predictions"][index]],
                        [batch_result["audio_length_s"][index]],
                        [batch_result["transcription_time_s"][index]],
                        append=True,
                    )
    else:
        warmup_dataset = prepare_dataset(args)
        dataset = warmup_dataset

        if args.skip_eval_samples is not None and args.skip_eval_samples > 0:
            print(f"Skipping first {args.skip_eval_samples} samples!")
            if args.streaming:
                dataset = dataset.skip(args.skip_eval_samples)
            else:
                if args.skip_eval_samples >= len(dataset):
                    raise RuntimeError(
                        f"skip_eval_samples={args.skip_eval_samples} leaves no samples "
                        f"for {args.dataset}:{args.split}."
                    )
                dataset = dataset.select(range(args.skip_eval_samples, len(dataset)))

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

        result_iter = iter(dataset)
        for result in tqdm(result_iter, desc="Samples..."):
            for key in all_results:
                all_results[key].append(result[key])
            if args.incremental_manifest:
                write_manifest_records(
                    manifest_path,
                    [result["references"]],
                    [result["predictions"]],
                    [result["audio_length_s"]],
                    [result["transcription_time_s"]],
                    append=True,
                )

    if not all_results["references"]:
        raise RuntimeError(
            f"No evaluation samples found for {args.dataset}:{args.split}. "
            "Check the dataset config/files and filtering before writing a manifest."
        )

    if args.incremental_manifest:
        write_manifest_records(
            manifest_path,
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
        )
    elif args.manifest_suffix:
        manifest_path = build_manifest_path(
            args.model_id,
            args.dataset_path,
            args.dataset,
            args.split,
            suffix=args.manifest_suffix,
        )
        write_manifest_records(
            manifest_path,
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
        )
    else:
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
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="ARK-ASR model path or Hugging Face repo id.")
    parser.add_argument("--processor_id", type=str, default=None, help="Processor path or repo id. Defaults to model_id.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/esb-datasets-test-only-sorted",
        help="Dataset path. By default, it is `hf-audio/esb-datasets-test-only-sorted`.",
    )
    parser.add_argument("--dataset_revision", type=str, default=None, help="Optional Hugging Face dataset revision.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument(
        "--local_parquet_dir",
        type=str,
        default=None,
        help="Optional local directory containing parquet shards for this dataset split.",
    )
    parser.add_argument("--device", type=int, default=-1, help="Device index. Use -1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples per batch.")
    parser.add_argument("--skip_eval_samples", type=int, default=0, help="Number of prepared evaluation samples to skip.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Number of samples to evaluate.")
    parser.add_argument("--manifest_suffix", type=str, default=None, help="Optional suffix for writing a partial manifest.")
    parser.add_argument(
        "--incremental_manifest",
        action="store_true",
        help="Write the manifest as samples are produced, then rewrite it once at the end.",
    )
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable dataset streaming.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warm-up batches.")
    parser.add_argument("--target_sr", type=int, default=16000, help="Evaluation sampling rate.")
    parser.add_argument(
        "--audio_decode",
        choices=["soundfile", "datasets"],
        default="datasets",
        help="Decode audio with soundfile or the default datasets.Audio decoder.",
    )
    parser.add_argument("--max_audio_seconds", type=int, default=40, help="Audio truncation length passed to the processor.")
    parser.add_argument(
        "--audio_input",
        choices=["temp_wav", "array"],
        default="array",
        help="Pass audio through temporary 16 kHz WAV files or directly as arrays.",
    )
    parser.add_argument("--audio_gain", type=float, default=1.0)
    parser.add_argument("--asr_block_token_id_from", type=int, default=151670)
    parser.add_argument("--padding_side", choices=["left", "right"], default="left")
    parser.add_argument("--attn_impl", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--force_clean_exit",
        action="store_true",
        help="Exit with os._exit(0) after successful evaluation to bypass interpreter shutdown issues in some stacks.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
    if args.force_clean_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
