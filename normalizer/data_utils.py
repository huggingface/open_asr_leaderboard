import os
import re
from functools import lru_cache

import num2words
from datasets import Audio, IterableDataset, load_dataset
from huggingface_hub import snapshot_download
from normalizer import (
    ArmenianTextNormalizer,
    BasicMultilingualTextNormalizer,
    EnglishTextNormalizer,
)

from .eval_utils import (
    merge_chunked_manifest,
    normalize_compound_pairs,
    read_manifest,
    write_manifest,
)


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


# Language-specific filler words / hesitations, removed when the matching
# lang= is passed to the multilingual normalizer. Multi-word entries are
# supported (matched on whitespace boundaries). Currently empty
FILLER_WORDS = {}

ARMENIAN_FLEURS_CONFIGS = {"fleurs_hy", "hy_am"}
ARMENIAN_FLEURS_CORRECTIONS_DATASET = "Metric-AI/fleurs-corrections"
MULTILINGUAL_DATASET_OVERRIDES = {
    "fleurs_hy": ("google/fleurs", "hy_am"),
    "mcv26_hy": ("deepdml/common_voice_26_0", "hy-AM"),
}


def resolve_multilingual_dataset(dataset_path, config_name):
    """Resolve a leaderboard config to its public upstream dataset."""
    return MULTILINGUAL_DATASET_OVERRIDES.get(
        config_name, (dataset_path, config_name)
    )


def load_multilingual_dataset(dataset_path, config_name, split, **kwargs):
    """Load a benchmark dataset and expose the common runner schema.

    Armenian datasets currently live outside the aggregate leaderboard dataset,
    so their canonical leaderboard config names are redirected here.  Keeping
    the canonical name in callers also keeps result filenames stable.
    """
    resolved_path, resolved_config = resolve_multilingual_dataset(
        dataset_path, config_name
    )
    if resolved_path == "deepdml/common_voice_26_0":
        kwargs.setdefault("trust_remote_code", True)
    dataset = load_dataset(
        resolved_path,
        resolved_config or None,
        split=split,
        **kwargs,
    )

    column_names = dataset.column_names
    if isinstance(column_names, dict):
        column_names = next(iter(column_names.values()), [])

    transcript_column = next(
        (
            name
            for name in ("text", "raw_transcription", "transcription", "sentence")
            if name in column_names
        ),
        None,
    )
    if transcript_column is None:
        raise ValueError(
            f"Dataset {resolved_path!r}/{resolved_config!r} has no supported transcript column"
        )
    if transcript_column != "text":
        dataset = dataset.map(lambda sample: {"text": sample[transcript_column]})

    # google/fleurs exposes the stable WAV name as `path`; the correction
    # manifest calls the same value `file_name`.
    if config_name in ARMENIAN_FLEURS_CONFIGS and "file_name" not in column_names:
        if "path" not in column_names:
            raise ValueError(
                "Armenian FLEURS has neither 'file_name' nor 'path' for corrections"
            )
        dataset = dataset.map(
            lambda sample: {"file_name": os.path.basename(sample["path"])}
        )

    # Common Voice 26 exposes the source filename as `path`; runners use the
    # leaderboard's canonical `file_name` field when creating local WAV files.
    if config_name == "mcv26_hy" and "file_name" not in column_names:
        dataset = dataset.map(
            lambda sample: {"file_name": os.path.basename(sample["path"])}
        )

    if (
        config_name in ARMENIAN_FLEURS_CONFIGS
        and split == "test"
        and not isinstance(dataset, IterableDataset)
    ):
        source_file_names = {
            os.path.basename(file_name) for file_name in dataset["file_name"]
        }
        missing = set(_armenian_fleurs_corrections()) - source_file_names
        if missing:
            examples = ", ".join(repr(name) for name in sorted(missing)[:3])
            raise ValueError(
                f"{len(missing)} Armenian FLEURS correction rows do not match "
                f"the evaluation dataset (examples: {examples})"
            )

    return apply_reference_corrections(dataset, config_name, split)


@lru_cache(maxsize=1)
def _armenian_fleurs_corrections():
    """Load the small, text-only correction manifest once per process."""
    corrections = load_dataset(
        ARMENIAN_FLEURS_CORRECTIONS_DATASET,
        split="test",
        token=True,
    )
    by_file_name = {}
    for row in corrections:
        file_name = os.path.basename(row["file_name"])
        correction = (
            row["original_transcription"],
            row["corrected_transcription"],
        )
        if file_name in by_file_name and by_file_name[file_name] != correction:
            raise ValueError(
                f"Conflicting Armenian FLEURS corrections for {file_name!r}"
            )
        by_file_name[file_name] = correction
    return by_file_name


def corrected_reference(sample, reference, config_name, split="test"):
    """Return the reviewed Armenian FLEURS reference for one dataset sample.

    Corrections are keyed by the stable FLEURS WAV filename.  An exact check
    against the recorded original transcript prevents a correction from being
    silently applied to the wrong dataset revision.
    """
    if config_name not in ARMENIAN_FLEURS_CONFIGS or split != "test":
        return reference

    file_name = sample.get("file_name") or sample.get("path")
    if not file_name:
        raise ValueError(
            "Armenian FLEURS samples must include 'file_name' so reviewed "
            "reference corrections can be joined safely"
        )
    correction = _armenian_fleurs_corrections().get(os.path.basename(file_name))
    if correction is None:
        return reference

    original, corrected = correction
    if reference not in {original, corrected}:
        raise ValueError(
            f"Armenian FLEURS source reference mismatch for {file_name!r}; "
            "the correction manifest and evaluation dataset may use different revisions"
        )
    return corrected


def apply_reference_corrections(dataset, config_name, split="test"):
    """Overlay reviewed references onto an Armenian FLEURS Dataset."""
    if config_name not in ARMENIAN_FLEURS_CONFIGS or split != "test":
        return dataset

    column_names = dataset.column_names
    if isinstance(column_names, dict):
        column_names = next(iter(column_names.values()), [])
    text_column = next(
        (
            name
            for name in ("text", "raw_transcription", "transcription", "sentence")
            if name in column_names
        ),
        None,
    )
    if text_column is None:
        raise ValueError(
            "Armenian FLEURS dataset has no supported transcript column"
        )
    if "file_name" not in column_names:
        raise ValueError(
            "Armenian FLEURS dataset has no 'file_name' column for corrections"
        )

    def apply(sample):
        sample[text_column] = corrected_reference(
            sample, sample[text_column], config_name, split
        )
        # All multilingual runners consume the canonical `text` column.
        sample["text"] = sample[text_column]
        return sample

    return dataset.map(apply)


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Call with just text for standard normalization (backward-compatible).
    Pass lang= to also convert digits to words via num2words and remove
    language-specific filler words (see FILLER_WORDS).
    """

    def __init__(self, remove_diacritics: bool = True):
        super().__init__(remove_diacritics)
        self._language_normalizers = {"hy": ArmenianTextNormalizer()}
        # Pre-compile filler patterns. Each filler word is passed through the
        # base normalization itself, so the pattern matches the normalized
        # text exactly (base normalization may strip punctuation such as "…"
        # or combining marks). Longest-first so that multi-word and longer
        # variants match before their prefixes. Matched on whitespace
        # boundaries ((?<!\S) / (?!\S)) rather than \b, which is unreliable
        # next to combining marks.
        self._filler_patterns = {}
        base_normalize = super().__call__
        for lang, words in FILLER_WORDS.items():
            normalized_words = {base_normalize(w) for w in words}
            normalized_words.discard("")
            self._filler_patterns[lang] = re.compile(
                r"(?<!\S)(?:"
                + "|".join(re.escape(w) for w in sorted(normalized_words, key=len, reverse=True))
                + r")(?!\S)"
            )

    def _remove_fillers(self, text, lang):
        pattern = self._filler_patterns.get(lang)
        if pattern is None:
            return text
        text = pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

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
        language_key = lang.lower().replace("_", "-") if lang is not None else None
        language_normalizer = self._language_normalizers.get(language_key)
        if language_normalizer is not None:
            return language_normalizer(s)

        s = super().__call__(s)
        if lang is not None:
            s = self._remove_fillers(s, lang)
            s = self._normalize_numbers(s, lang)
        return s


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "raw_transcription" in sample:
        return sample["raw_transcription"]
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
            f"Expected transcript column of either 'text', 'raw_transcription', 'sentence', "
            f"'normalized_text', 'transcript', or 'transcription'. Got sample keys: "
            f"{list(sample.keys())}. Ensure a text column name is present in the dataset."
        )


normalizer = EnglishTextNormalizer()

ml_normalizer = MultilingualNormalizer(remove_diacritics=False)


def normalize(batch):
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = normalizer(batch["original_text"])
    return batch


# Chunked datasets provide one reference transcript per parent session instead of
# one per chunk. Maps the chunked dataset repo to the repo holding the transcripts.
CHUNKED_DATASETS = {
    "artificialanalysis/earnings22-cleaned-aa-chunked": {
        "parent_dataset_path": "ArtificialAnalysis/Earnings22-Cleaned-AA",
        "audio_dir": "audio",
    },
}

# Carried into the results manifest so chunks can be reassembled at scoring time.
CHUNK_METADATA_KEYS = ["parent_id", "chunk_index"]


def is_chunked_dataset(dataset_path):
    return str(dataset_path).lower() in CHUNKED_DATASETS


def load_chunked_data(args):
    """Load a chunked dataset, attaching each chunk's parent transcript as `text`."""
    config = CHUNKED_DATASETS[str(args.dataset_path).lower()]

    parent_dataset = load_dataset(
        config["parent_dataset_path"], split=args.split, token=True
    )
    parent_text = {sample["id"]: get_text(sample) for sample in parent_dataset}

    dataset = load_dataset(args.dataset_path, split=args.split, token=True)

    # Audio is not referenced by the metadata file, so fetch it separately.
    audio_root = snapshot_download(
        repo_id=args.dataset_path,
        repo_type="dataset",
        allow_patterns=[f"{config['audio_dir']}/*"],
    )

    missing = sorted(set(dataset["parent_id"]) - set(parent_text))
    if missing:
        raise ValueError(
            f"No transcript found in {config['parent_dataset_path']} for parent "
            f"id(s) {missing}. The chunked and parent datasets are out of sync."
        )

    def attach_audio_and_text(sample):
        sample["audio"] = os.path.join(
            audio_root, config["audio_dir"], sample["file_name"]
        )
        sample["text"] = parent_text[sample["parent_id"]]
        return sample

    dataset = dataset.map(attach_audio_and_text, load_from_cache_file=False)
    dataset = dataset.cast_column("audio", Audio())

    return dataset


def load_data(args):
    if is_chunked_dataset(args.dataset_path):
        return load_chunked_data(args)

    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )

    return dataset


def prepare_data(dataset, sampling_rate=16000):
    # Re-sample and normalize transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    # NOTE (ebezzam) don't load from cache to account for potential changes in normalization logic
    # IterableDataset (streaming) has no cache, so the kwarg is only needed for Dataset
    map_kwargs = (
        {} if isinstance(dataset, IterableDataset) else {"load_from_cache_file": False}
    )
    dataset = dataset.map(normalize, **map_kwargs)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    return dataset


AUDIO_FILEPATH_METADATA_KEYS = [
    "id",  # Main: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard
    "file_name",  # Multilingual: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-multilingual-datasets
    "file_name",  # Private
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
