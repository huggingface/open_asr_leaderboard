"""Dataset selections shared by download, inference and local scoring."""

import shlex

DEFAULT_DATASET_PATH = "hf-audio/open-asr-leaderboard"

LANGUAGE_NAMES = {
    # Note: it supports more languages
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
}


def parse_datasets(spec, default_path=DEFAULT_DATASET_PATH):
    selections = []
    seen = set()
    for line in spec.splitlines():
        if not line.strip():
            continue
        fields = shlex.split(line)
        if not 2 <= len(fields) <= 5:
            raise ValueError(
                f"Expected 'name split [dataset_path [config [language]]]': {line}"
            )
        name, split = fields[:2]
        path = fields[2] if len(fields) >= 3 else default_path
        config = fields[3] if len(fields) >= 4 else (name if len(fields) == 2 else None)
        config = None if config == "-" else config
        language = fields[4] if len(fields) == 5 else "en"
        if language not in LANGUAGE_NAMES:
            raise ValueError(f"Unsupported language: {language}")
        identity = (path, config, split)
        if identity in seen:
            raise ValueError(f"Duplicate dataset selection: {line}")
        seen.add(identity)
        selections.append((*identity, language))
    if not selections:
        raise ValueError("Select at least one dataset")
    return selections


def manifest_name(model_id, path, config, split):
    # Match the shared normalizer's write_manifest naming.
    model_id, path, config = (
        value.replace("/", "-") for value in (model_id, path, config or "")
    )
    return f"MODEL_{model_id}_DATASET_{path}_{config}_{split}.jsonl"
