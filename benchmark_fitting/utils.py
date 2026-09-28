"""Helpers shared by the scorers in this folder.

Both scorers read the same published prediction manifests, so locating them,
parsing them, naming models and syncing the bucket is one problem solved once
here rather than twice. What is genuinely per-scorer is passed in: which rows
count as usable, how a filename's dataset group maps to a dataset, and how to
describe the model set when ``--model`` matches nothing.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "DEFAULT_BUCKET",
    "BUCKET_RE",
    "read_manifest",
    "find_manifests",
    "canonical_model",
    "resolve_model",
    "sync_bucket",
]

DEFAULT_BUCKET = "hf-audio/asr_leaderboard_h200"

# Manifest names in the results bucket, e.g.
# MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_test.jsonl
BUCKET_RE = re.compile(r"^MODEL_(?P<model>.+?)_DATASET_(?P<dataset>.+)\.jsonl$")


def read_manifest(path: str, required: Iterable[str] = ()) -> list[dict]:
    """Rows of a manifest, skipping any that lack a field in ``required``.

    Contributed manifests are not uniformly well-formed; one bad row must not
    abort a dataset that takes minutes to score. With no ``required`` fields
    every JSON object is kept, and only unparseable or non-object rows are
    dropped.
    """
    out = []
    malformed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict) and all(isinstance(row.get(k), str) for k in required):
                out.append(row)
            else:
                malformed += 1
    if malformed:
        print(f"  note: skipped {malformed} malformed rows in {path}")
    return out


def find_manifests(
    root: str, dataset: str, dataset_of: Callable[[str], str | None] | None = None
) -> dict[str, str]:
    """Map model name to its manifest for ``dataset``, under ``root``.

    Handles both the results-bucket layout (``<model>/MODEL_<model>_DATASET_<...>_
    <dataset>.jsonl``) and a flat per-dataset cache (``<dataset>/<model>.jsonl``).

    ``dataset_of`` maps the dataset group of a bucket filename to the dataset it
    belongs to; by default any group ending in ``dataset`` counts. Pass one to
    resolve a group against a known set of tags instead, so that a longer tag
    wins over a shorter suffix of it.
    """
    if dataset_of is None:

        def dataset_of(group: str) -> str | None:
            return dataset if group.endswith(dataset) else None

    out: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        base = os.path.basename(path)
        match = BUCKET_RE.match(base)
        if match:
            if dataset_of(match.group("dataset")) != dataset:
                continue
            model = match.group("model")
        else:
            if os.path.basename(os.path.dirname(path)) != dataset:
                continue
            model = base[: -len(".jsonl")]
        if model in out:
            print(f"  note: ignoring duplicate manifest for {model}: {path}")
            continue
        out[model] = path
    return out


def canonical_model(name: str) -> str:
    """A model name in the form the results bucket uses.

    Manifest filenames cannot carry the ``/`` of a Hub id, so the bucket writes
    ``openai-whisper-large-v3`` for ``openai/whisper-large-v3``. Folding the
    separator and case here lets either form be given on the command line.
    :func:`normalizer.to_hub_ids` is the inverse, used when writing the outputs.
    """
    return name.replace("/", "-").lower()


def resolve_model(
    name: str,
    available: dict[str, str],
    skipped: Sequence[tuple[str, str]] = (),
    what: str = "model with a manifest",
) -> str:
    """Match ``name`` against the models in ``available``.

    Exact match first, then a unique substring match, both up to
    :func:`canonical_model`, so a model can be named by its Hub id, its bucket
    id, or an abbreviation of either. Anything else is an error: silently
    scoring the wrong model is worse than stopping. ``skipped`` names models
    that were dropped earlier with a reason, so that matching one of those
    reports why rather than claiming the model does not exist.
    """
    if name in available:
        return name
    query = canonical_model(name)
    exact = sorted(model for model in available if canonical_model(model) == query)
    if exact:
        return exact[0]
    matches = sorted(model for model in available if query in canonical_model(model))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listed = "\n  ".join(matches)
        sys.exit(f"--model {name!r} matches {len(matches)} models:\n  {listed}")
    for model, why in skipped:
        if query in canonical_model(model):
            sys.exit(f"--model {name!r} matches {model}, which has no usable hypotheses: {why}")
    listed = "\n  ".join(sorted(available))
    sys.exit(f"--model {name!r} matches no {what}. Available:\n  {listed}")


def sync_bucket(bucket: str, local_dir: str, hf_token: str | None = None) -> None:
    """Sync an HF bucket to a local directory using the `hf` CLI."""
    bucket_url = f"hf://buckets/{bucket}"
    print(f"Syncing {bucket_url}  →  {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    subprocess.run(["hf", "buckets", "sync", bucket_url, local_dir], check=True, env=env)
    print("Sync complete.\n")
