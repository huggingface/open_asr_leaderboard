import re
import os

import num2words
from datasets import load_dataset, Audio
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer

from .eval_utils import read_manifest, write_manifest, normalize_compound_pairs


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Call with just text for standard normalization (backward-compatible).
    Pass lang= to also convert digits to words via num2words.
    """

    def _normalize_numbers(self, text, lang):
        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)
        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()
        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        s = super().__call__(s)
        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )

normalizer = EnglishTextNormalizer()

ml_normalizer = MultilingualNormalizer(remove_diacritics=False)


def normalize(batch):
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = normalizer(batch["original_text"])
    return batch


def load_data(args):
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )

    return dataset

def prepare_data(dataset, sampling_rate=16000):
    # Re-sample and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    dataset = dataset.map(normalize)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    return dataset


AUDIO_FILEPATH_METADATA_KEYS = [
    "id",           # Main: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard
    "file_name",    # Multilingual: https://huggingface.co/datasets/nithinraok/asr-leaderboard-datasets
    "file_name",    # Private
]


def _basename_or_none(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return os.path.basename(value)


def extract_audio_filepath_from_sample(sample):
    if sample is None:
        return None

    for key in AUDIO_FILEPATH_METADATA_KEYS:
        try:
            if key in sample:
                basename = _basename_or_none(sample[key])
                if basename is not None:
                    return basename
        except TypeError:
            # AudioDecoder / other non-mapping sample types are not subscriptable.
            return None
    return None


def extract_audio_filepaths_from_batch(batch, batch_size=None):
    if batch_size is None:
        if "audio" in batch:
            batch_size = len(batch["audio"])
        elif len(batch) > 0:
            first_value = next(iter(batch.values()))
            if isinstance(first_value, list):
                batch_size = len(first_value)

    if batch_size is None:
        return []

    for key in AUDIO_FILEPATH_METADATA_KEYS:
        values = batch.get(key)
        if isinstance(values, list) and len(values) == batch_size:
            return [_basename_or_none(v) for v in values]
    return [None] * batch_size
