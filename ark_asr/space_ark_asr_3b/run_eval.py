import argparse
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from kaldialign import edit_distance as kaldi_edit_distance
from normalizer import data_utils
from torch import Tensor
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)"
    r")\|>"
)
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")
CONTROL_TOKEN_PATTERN = re.compile(r"^<.*>$")
AUDIO_FILEPATH_METADATA_KEYS = ("id", "file_name", "path")
STREAMING_PAD_TOKEN = "[STREAMING_PAD]"
STREAMING_WORD_TOKEN = "[STREAMING_WORD]"


@dataclass(frozen=True)
class ArkStreamingAudioConfig:
    sampling_rate: int = 16000
    frame_rate: float = 12.5
    streaming_n_left_pad_tokens: int = 32

    @property
    def raw_audio_samples_per_token(self) -> int:
        samples = self.sampling_rate / self.frame_rate
        if not samples.is_integer():
            raise ValueError(
                "sampling_rate / frame_rate must be an integer: "
                f"sampling_rate={self.sampling_rate} frame_rate={self.frame_rate}"
            )
        return int(samples)


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


def resolve_required_token_id(tokenizer: Any, *, attr_name: str | None, token: str) -> int:
    if attr_name:
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            return int(token_id)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None and int(token_id) >= 0:
        return int(token_id)
    raise ValueError(f"Tokenizer cannot resolve required token id for {token!r}.")


def resolve_ark_streaming_special_token_ids(tokenizer: Any) -> dict[str, int]:
    return {
        "bos_token_id": resolve_required_token_id(tokenizer, attr_name="bos_token_id", token="<|im_start|>"),
        "eos_token_id": resolve_required_token_id(tokenizer, attr_name="eos_token_id", token="<|im_end|>"),
        "pad_token_id": resolve_required_token_id(tokenizer, attr_name="pad_token_id", token="<|endoftext|>"),
        "streaming_pad_token_id": resolve_required_token_id(
            tokenizer, attr_name=None, token=STREAMING_PAD_TOKEN
        ),
        "streaming_word_token_id": resolve_required_token_id(
            tokenizer, attr_name=None, token=STREAMING_WORD_TOKEN
        ),
    }


def load_ark_streaming_audio_config(model_id: str) -> ArkStreamingAudioConfig:
    config_path = Path(model_id) / "config.json"
    if not config_path.is_file():
        return ArkStreamingAudioConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    audio_config = config.get("audio_config")
    if not isinstance(audio_config, dict):
        return ArkStreamingAudioConfig()
    return ArkStreamingAudioConfig(
        sampling_rate=int(audio_config.get("sampling_rate", ArkStreamingAudioConfig.sampling_rate)),
        frame_rate=float(audio_config.get("frame_rate", ArkStreamingAudioConfig.frame_rate)),
        streaming_n_left_pad_tokens=int(
            audio_config.get(
                "streaming_n_left_pad_tokens",
                ArkStreamingAudioConfig.streaming_n_left_pad_tokens,
            )
        ),
    )


def is_ark_streaming_asr_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    return (
        getattr(config, "model_type", None) == "ark_streaming_asr"
        and hasattr(model, "get_audio_features")
        and hasattr(model, "language_model")
        and hasattr(model, "delay_embedding")
    )


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


def clean_prediction_text(text: str) -> str:
    if not text:
        return ""
    text = truncate_generation_text(text)
    text = remove_special_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def compute_wer_with_compounds(references: list[str], predictions: list[str]) -> float:
    total_ins = total_del = total_sub = total_ref_words = 0
    for ref, pred in zip(references, predictions):
        ref_words = ref.split()
        pred_words = pred.split()
        if not ref_words:
            total_ins += len(pred_words)
            continue
        result = kaldi_edit_distance(ref_words, pred_words, merge_compounds=True)
        total_ins += result["ins"]
        total_del += result["del"]
        total_sub += result["sub"]
        total_ref_words += result["ref_len"]

    total_errors = total_ins + total_del + total_sub
    return total_errors / total_ref_words if total_ref_words > 0 else 0.0


def _basename_or_none(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return os.path.basename(value)


def extract_audio_filepath_from_sample(sample: dict[str, Any]) -> str | None:
    if hasattr(data_utils, "extract_audio_filepath_from_sample"):
        return data_utils.extract_audio_filepath_from_sample(sample)

    for key in AUDIO_FILEPATH_METADATA_KEYS:
        if key in sample:
            basename = _basename_or_none(sample[key])
            if basename is not None:
                return basename

    audio = sample.get("audio")
    if isinstance(audio, dict):
        return _basename_or_none(audio.get("path"))
    return None


def extract_audio_from_local_parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    audio = row.get("audio")
    if isinstance(audio, dict):
        return audio

    audio_bytes = row.get("bytes")
    audio_path = row.get("path")
    if audio_bytes is None and not audio_path:
        raise ValueError(f"Local parquet row must contain audio, bytes, or path. Got keys: {list(row)}")
    return {"bytes": audio_bytes, "path": audio_path}


def extract_audio_filepaths_from_batch(batch: dict[str, Any], batch_size: int) -> list[str | None]:
    if hasattr(data_utils, "extract_audio_filepaths_from_batch"):
        return data_utils.extract_audio_filepaths_from_batch(batch, batch_size)

    for key in AUDIO_FILEPATH_METADATA_KEYS:
        values = batch.get(key)
        if isinstance(values, list) and len(values) == batch_size:
            return [_basename_or_none(value) for value in values]

    audios = batch.get("audio")
    if isinstance(audios, list) and len(audios) == batch_size:
        return [
            _basename_or_none(audio.get("path")) if isinstance(audio, dict) else None
            for audio in audios
        ]
    return [None] * batch_size


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


def limit_cuda_memory(device: str, limit_gb: float | None) -> None:
    if limit_gb is None or limit_gb <= 0 or not device.startswith("cuda"):
        return
    if ":" in device:
        device_index = int(device.split(":", 1)[1])
    else:
        device_index = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    fraction = min(float(limit_gb) * (1024 ** 3) / total_memory, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def load_model(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_impl: str,
    revision: str | None,
):
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
                revision=revision,
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
            token=data_utils.get_hf_token(),
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
    batch = {"audio": [], "norm_text": [], "original_text": [], "audio_filepath": []}
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for row_group_index in range(pf.num_row_groups):
            rows = pf.read_row_group(row_group_index).to_pylist()
            for row in rows:
                original_text = data_utils.get_text(row)
                normalized_text = data_utils.normalizer(original_text)
                if not data_utils.is_target_text_in_range(normalized_text):
                    continue
                if seen < skip_samples:
                    seen += 1
                    continue
                if max_samples is not None and max_samples > 0 and yielded >= max_samples:
                    if batch["audio"]:
                        yield batch
                    return

                batch["audio"].append(extract_audio_from_local_parquet_row(row))
                batch["norm_text"].append(normalized_text)
                batch["original_text"].append(original_text)
                batch["audio_filepath"].append(extract_audio_filepath_from_sample(row))
                yielded += 1
                seen += 1
                if len(batch["audio"]) >= batch_size:
                    yield batch
                    batch = {"audio": [], "norm_text": [], "original_text": [], "audio_filepath": []}

    if batch["audio"]:
        yield batch


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


def pad_audio_to_min_seconds(audio: np.ndarray, sampling_rate: int, min_seconds: float) -> np.ndarray:
    if min_seconds <= 0:
        return audio
    min_samples = int(round(min_seconds * sampling_rate))
    if len(audio) >= min_samples:
        return audio
    return np.pad(audio, (0, min_samples - len(audio)), mode="constant")


def build_ark_offline_audio_timeline(
    audio: np.ndarray,
    *,
    audio_config: ArkStreamingAudioConfig,
    num_delay_tokens: int,
    right_pad_text_tokens: int,
) -> tuple[np.ndarray, dict[str, int]]:
    samples_per_token = int(audio_config.raw_audio_samples_per_token)
    left_pad_tokens = int(audio_config.streaming_n_left_pad_tokens)
    real_audio_tokens = max(1, math.ceil(int(audio.shape[0]) / samples_per_token))
    padded_real_samples = real_audio_tokens * samples_per_token
    if audio.shape[0] < padded_real_samples:
        audio = np.pad(audio, (0, padded_real_samples - audio.shape[0]), mode="constant")

    right_pad_tokens = int(num_delay_tokens) + 1 + int(right_pad_text_tokens)
    left_silence = np.zeros(left_pad_tokens * samples_per_token, dtype=np.float32)
    right_silence = np.zeros(right_pad_tokens * samples_per_token, dtype=np.float32)
    padded_audio = np.concatenate([left_silence, audio.astype(np.float32, copy=False), right_silence])
    return padded_audio, {
        "left_pad_tokens": left_pad_tokens,
        "real_audio_tokens": real_audio_tokens,
        "right_pad_tokens": right_pad_tokens,
        "total_token_count": left_pad_tokens + real_audio_tokens + right_pad_tokens,
        "num_delay_tokens": int(num_delay_tokens),
    }


def decode_ark_visible_text(
    tokenizer: Any,
    generated_token_ids: list[int],
    special_ids: dict[str, int],
) -> str:
    visible_token_ids: list[int] = []
    for token_id in generated_token_ids:
        if token_id in {
            special_ids["streaming_pad_token_id"],
            special_ids["streaming_word_token_id"],
            special_ids["bos_token_id"],
            special_ids["pad_token_id"],
        }:
            continue
        if token_id == special_ids["eos_token_id"]:
            break
        visible_token_ids.append(int(token_id))
    return tokenizer.decode(
        visible_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def ark_offline_greedy_decode(
    *,
    model: Any,
    tokenizer: Any,
    feature_extractor: Any,
    audios: list[np.ndarray],
    audio_config: ArkStreamingAudioConfig,
    dtype: torch.dtype,
    device: str,
    max_new_tokens: int,
    max_audio_seconds: int,
    audio_gain: float,
    right_pad_text_tokens: int,
) -> list[str]:
    special_ids = resolve_ark_streaming_special_token_ids(tokenizer)
    num_delay_tokens = int(getattr(model.config, "default_num_delay_tokens", 6))
    clipped_audios = []
    timelines = []
    max_audio_samples = int(max_audio_seconds * audio_config.sampling_rate)
    for audio in audios:
        audio = apply_audio_gain(torch.from_numpy(audio), audio_gain).cpu().numpy()
        if max_audio_samples > 0:
            audio = audio[:max_audio_samples]
        padded_audio, timeline = build_ark_offline_audio_timeline(
            audio,
            audio_config=audio_config,
            num_delay_tokens=num_delay_tokens,
            right_pad_text_tokens=right_pad_text_tokens,
        )
        clipped_audios.append(padded_audio)
        timelines.append(timeline)

    feature_batch = feature_extractor(
        clipped_audios,
        sampling_rate=audio_config.sampling_rate,
        padding="longest",
        return_tensors="pt",
        center=True,
    )
    input_features = feature_batch["input_features"].to(device=device, dtype=dtype)

    with torch.inference_mode():
        audio_embeds = model.get_audio_features(input_features=input_features).to(device=device, dtype=dtype)
        token_embedder = model.language_model.get_input_embeddings()
        delay_ids = torch.full((len(audios),), num_delay_tokens, device=device, dtype=torch.long)
        delay_embed = model.delay_embedding(delay_ids).to(dtype=dtype).unsqueeze(1)
        generated_per_sample: list[list[int]] = []

        for sample_index, timeline in enumerate(timelines):
            left_pad_tokens = int(timeline["left_pad_tokens"])
            total_token_count = int(timeline["total_token_count"])
            if int(audio_embeds.shape[1]) < total_token_count:
                raise ValueError(
                    "Audio embedding token count is shorter than timeline: "
                    f"audio_embeds={tuple(audio_embeds.shape)} total_token_count={total_token_count}"
                )

            prompt_token_ids = [special_ids["bos_token_id"]]
            prompt_token_ids.extend([special_ids["streaming_pad_token_id"]] * max(0, left_pad_tokens - 1))
            input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
            inputs_embeds = token_embedder(input_ids).to(dtype=dtype)
            sample_audio_embeds = audio_embeds[sample_index : sample_index + 1, :, :]
            sample_delay_embed = delay_embed[sample_index : sample_index + 1, :, :]
            inputs_embeds = inputs_embeds + sample_audio_embeds[:, :left_pad_tokens, :] + sample_delay_embed
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((1, left_pad_tokens), device=device, dtype=torch.long),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            generated_token_ids = [next_token_id]
            last_token_id = next_token_id
            for token_index in range(left_pad_tokens, total_token_count):
                current_input_ids = torch.tensor([[last_token_id]], device=device, dtype=torch.long)
                step_embeds = token_embedder(current_input_ids).to(dtype=dtype)
                step_embeds = (
                    step_embeds
                    + sample_audio_embeds[:, token_index : token_index + 1, :]
                    + sample_delay_embed
                )
                outputs = model.language_model(
                    inputs_embeds=step_embeds,
                    attention_mask=torch.ones((1, token_index + 1), device=device, dtype=torch.long),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                last_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
                generated_token_ids.append(last_token_id)
                if (
                    last_token_id == special_ids["eos_token_id"]
                    and token_index >= left_pad_tokens + num_delay_tokens
                ):
                    break
            generated_per_sample.append(generated_token_ids)

    return [
        clean_prediction_text(decode_ark_visible_text(tokenizer, token_ids, special_ids))
        for token_ids in generated_per_sample
    ]


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
        revision: str | None,
        padding_side: str,
        asr_block_token_id_from: int,
    ) -> None:
        self.device = device
        self.torch_dtype = resolve_torch_dtype(dtype, device)
        self.model, self.resolved_attn_impl = load_model(
            model_id,
            device,
            self.torch_dtype,
            attn_impl,
            revision,
        )
        self.model.config.use_cache = True
        if hasattr(self.model, "language_model"):
            self.model.language_model.config.use_cache = True
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = padding_side
        self.processor = AutoProcessor.from_pretrained(
            processor_id or model_id,
            trust_remote_code=True,
            revision=revision,
            fix_mistral_regex=True,
        )
        if hasattr(self.processor, "tokenizer"):
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.tokenizer.pad_token_id
            self.processor.tokenizer.padding_side = padding_side
        self.is_ark_streaming_asr = is_ark_streaming_asr_model(self.model)
        self.ark_feature_extractor = None
        self.ark_audio_config = None
        if self.is_ark_streaming_asr:
            self.ark_feature_extractor = AutoFeatureExtractor.from_pretrained(
                processor_id or model_id,
                trust_remote_code=True,
                revision=revision,
            )
            self.ark_audio_config = load_ark_streaming_audio_config(model_id)
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

        if self.is_ark_streaming_asr:
            decoded = [
                decode_audio(audio_input if isinstance(audio_input, dict) else {"path": audio_input}, target_sr)
                for audio_input in audio_inputs
            ]
            return ark_offline_greedy_decode(
                model=self.model,
                tokenizer=self.tokenizer,
                feature_extractor=self.ark_feature_extractor,
                audios=[audio for audio, _ in decoded],
                audio_config=self.ark_audio_config,
                dtype=self.torch_dtype,
                device=self.device,
                max_new_tokens=max_new_tokens,
                max_audio_seconds=max_audio_seconds,
                audio_gain=audio_gain,
                right_pad_text_tokens=10,
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
            predictions.append(clean_prediction_text(prediction_raw))
        return predictions


def main(args):
    device = resolve_device(args.device)
    limit_cuda_memory(device, args.gpu_memory_limit_gb)
    inferencer = ArkAsrInferencer(
        args.model_id,
        args.processor_id,
        device=device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        revision=args.revision,
        padding_side=args.padding_side,
        asr_block_token_id_from=args.asr_block_token_id_from,
    )
    print(f"Loaded {args.model_id}; device={device}; attn={inferencer.resolved_attn_impl}; dtype={inferencer.torch_dtype}")

    def benchmark(batch):
        decoded = [decode_audio(audio, args.target_sr) for audio in batch["audio"]]
        audios = [
            pad_audio_to_min_seconds(audio, sampling_rate, args.min_audio_seconds)
            for audio, sampling_rate in decoded
        ]
        sampling_rates = [sampling_rate for _, sampling_rate in decoded]
        batch["audio_length_s"] = [len(audio) / sampling_rate for audio, sampling_rate in zip(audios, sampling_rates)]
        minibatch_size = len(audios)
        batch["audio_filepath"] = extract_audio_filepaths_from_batch(batch, minibatch_size)

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
        batch["predictions"] = pred_text
        batch["references"] = batch["original_text"]
        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        if args.local_parquet_dir:
            print("Skipping warmup for local parquet input.")
        else:
            warmup_dataset = prepare_dataset(args)
            num_warmup_samples = args.warmup_steps * args.batch_size
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
            warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

            for _ in tqdm(warmup_dataset, desc="Warming up..."):
                continue

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }

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
    else:
        warmup_dataset = prepare_dataset(args)
        dataset = warmup_dataset

        if args.skip_eval_samples is not None and args.skip_eval_samples > 0:
            print(f"Skipping first {args.skip_eval_samples} samples!")
            if args.skip_eval_samples >= len(dataset):
                raise RuntimeError(
                    f"skip_eval_samples={args.skip_eval_samples} leaves no samples "
                    f"for {args.dataset}:{args.split}."
                )
            dataset = dataset.select(range(args.skip_eval_samples, len(dataset)))

        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
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

    if not all_results["references"]:
        raise RuntimeError(
            f"No evaluation samples found for {args.dataset}:{args.split}. "
            "Check the dataset config/files and filtering before writing a manifest."
        )

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.manifest_model_id or args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_refs = [data_utils.normalizer(ref) for ref in all_results["references"]]
    norm_preds = [data_utils.normalizer(pred) for pred in all_results["predictions"]]
    wer = round(100 * compute_wer_with_compounds(norm_refs, norm_preds), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="ARK-ASR model path or Hugging Face repo id.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use (e.g. 'refs/pr/11' for a PR branch). Defaults to the main branch.",
    )
    parser.add_argument(
        "--manifest_model_id",
        type=str,
        default=None,
        help="Optional short model id used only for manifest filenames and scoring.",
    )
    parser.add_argument("--processor_id", type=str, default=None, help="Processor path or repo id. Defaults to model_id.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/open-asr-leaderboard",
        help="Dataset path. By default, it is `hf-audio/open-asr-leaderboard`.",
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
    parser.add_argument("--gpu_memory_limit_gb", type=float, default=None, help="Optional per-process CUDA memory limit.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples per batch.")
    parser.add_argument("--skip_eval_samples", type=int, default=0, help="Number of prepared evaluation samples to skip.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Number of samples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warm-up batches.")
    parser.add_argument("--target_sr", type=int, default=16000, help="Evaluation sampling rate.")
    parser.add_argument("--min_audio_seconds", type=float, default=2.0, help="Right-pad audio shorter than this length.")
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

    main(args)
    if args.force_clean_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
