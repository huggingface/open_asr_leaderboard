#!/usr/bin/env python3
# Adapted from the reference implementation accompanying "Quantifying Benchmark
# Optimization in ASR Models" (https://github.com/tlebryk/asr-benchmark-optimization,
# Apache-2.0).
"""Score reference rendering agreement on the English short-form benchmarks.

The leaderboard's text normalizer rewrites part of every raw reference
transcript. This scorer keeps en-GB/en-US spelling, pointed acronyms, and
digits-versus-number-words: choices the audio does not determine and English WER
erases. Where a model's normalized output reproduces the span's words, the script
asks whether its raw output uses the reference's rendering.

One rate is written per model per independent English short-form dataset. Casing,
punctuation, and normalizer abbreviation rewrites are excluded.

Nothing is inferred and no audio is read: references and hypotheses both come
from the prediction manifests already published in the results bucket, scored
row by row within each manifest.

Usage:
    # Sync the public results bucket, then score every model in it.
    python ref_rendering/score_ref_rendering.py

    # Score an already-downloaded copy of the predictions.
    python ref_rendering/score_ref_rendering.py --preds_dir results

    # One dataset only.
    python ref_rendering/score_ref_rendering.py --preds_dir results --datasets voxpopuli_test

Writes `ref_rendering_<dataset>.csv` (one row per model) into `--out_dir`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

from ref_rendering_utils import REPORTED_CLASSES, score_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

DEFAULT_BUCKET = "hf-audio/asr_leaderboard_h200"

# The English short-form sets, whose references the English normalizer applies
# to. Longest match wins when a manifest name is matched against these.
SHORT_FORM_DATASETS = (
    "ami_test",
    "ami_cleaned_test",
    "earnings22_test",
    "gigaspeech_test",
    "gigaspeech_cleaned_test",
    "librispeech_test.clean",
    "librispeech_test.other",
    "spgispeech_test",
    "tedlium_test",
    "voxpopuli_test",
)

# Manifest names in the results bucket, e.g.
# MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_test.jsonl
BUCKET_RE = re.compile(r"^MODEL_(?P<model>.+?)_DATASET_(?P<dataset>.+)\.jsonl$")


# ---------------------------------------------------------------------------
# Reading predictions
# ---------------------------------------------------------------------------


def read_manifest(path: str) -> list[dict]:
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
            if isinstance(row, dict):
                out.append(row)
            else:
                malformed += 1
    if malformed:
        print(f"  note: skipped {malformed} malformed rows in {path}")
    return out


def _dataset_tag(name: str, datasets) -> str | None:
    """The dataset tag ``name`` ends with, longest first."""
    for tag in sorted(datasets, key=len, reverse=True):
        if name == tag or name.endswith("_" + tag) or name.endswith(tag):
            return tag
    return None


def discover_datasets(root: str) -> list[str]:
    """Short-form dataset tags with at least one manifest under ``root``."""
    found = set()
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        base = os.path.basename(path)
        match = BUCKET_RE.match(base)
        name = match.group("dataset") if match else os.path.basename(os.path.dirname(path))
        tag = _dataset_tag(name, SHORT_FORM_DATASETS)
        if tag:
            found.add(tag)
    return [tag for tag in SHORT_FORM_DATASETS if tag in found]


def find_manifests(root: str, dataset: str) -> dict[str, str]:
    """Map model name to its manifest for ``dataset``, under ``root``.

    Handles both the results-bucket layout (``<model>/MODEL_<model>_DATASET_<...>_
    <dataset>.jsonl``) and a flat per-dataset cache (``<dataset>/<model>.jsonl``).
    """
    out: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        base = os.path.basename(path)
        match = BUCKET_RE.match(base)
        if match:
            if _dataset_tag(match.group("dataset"), SHORT_FORM_DATASETS) != dataset:
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_dataset(manifests: dict[str, str]) -> list[dict]:
    """One row per model, sorted by descending agreement rate."""
    rows = []
    for model in sorted(manifests):
        pairs = []
        for row in read_manifest(manifests[model]):
            if "text" not in row or "pred_text" not in row:
                continue
            pairs.append((row["text"], row["pred_text"]))
        agg = score_pairs(pairs)
        scored = agg["scored"]
        # A model with no eligible span has no rate; an empty cell keeps it
        # distinguishable from a model that agreed at none of its spans.
        if scored[1] == 0:
            out = {"model": model, "rate": "", "n": 0}
        else:
            out = {"model": model, "rate": scored[0] / scored[1], "n": scored[1]}
        for cls in REPORTED_CLASSES:
            k, n = agg["by_class"][cls]
            out[f"{cls}_rate"] = k / n if n else 0.0
            out[f"{cls}_n"] = n
        rows.append(out)
    rows.sort(key=lambda r: (r["rate"] == "", -(r["rate"] or 0.0)))
    return rows


FIELDNAMES = ["model", "rate", "n"] + [
    f"{cls}_{suffix}" for cls in REPORTED_CLASSES for suffix in ("rate", "n")
]


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        for row in rows:
            # Counts and the model id stay verbatim; a missing rate stays empty.
            writer.writerow(
                [
                    row[k] if k == "model" or k == "n" or k.endswith("_n") or row[k] == "" else f"{row[k]:.6f}"
                    for k in FIELDNAMES
                ]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"HF results bucket. Default: {DEFAULT_BUCKET}")
    parser.add_argument(
        "--preds_dir",
        default=None,
        help="Read predictions from this directory instead of syncing the bucket.",
    )
    parser.add_argument(
        "--local_dir",
        default=os.path.join(REPO_ROOT, "results"),
        help="Directory to sync the bucket into. Default: <repo_root>/results",
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN.")
    default_out = os.path.join(REPO_ROOT, "results")
    parser.add_argument("--out_dir", default=default_out, help=f"Where to write the outputs. Default: {default_out}")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset tags. Default: every English short-form set found.",
    )
    args = parser.parse_args()

    if args.preds_dir:
        root = args.preds_dir
        print(f"Reading predictions from {root}\n")
    else:
        root = args.local_dir
        sync_bucket(args.bucket, root, hf_token=args.hf_token)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        # An unrecognised tag matches no manifest and would be reported as an empty
        # dataset rather than as the typo it is.
        unknown = [d for d in datasets if d not in SHORT_FORM_DATASETS]
        if unknown:
            sys.exit(f"Unknown dataset tags: {', '.join(unknown)}. Known: {', '.join(SHORT_FORM_DATASETS)}")
    else:
        datasets = discover_datasets(root)
        print(f"datasets found: {', '.join(datasets) or 'none'}")
    if not datasets:
        sys.exit("No short-form dataset manifests found; nothing to score.")

    os.makedirs(args.out_dir, exist_ok=True)
    for dataset in datasets:
        manifests = find_manifests(root, dataset)
        print(f"\n=== {dataset}: {len(manifests)} manifests")
        if not manifests:
            continue
        rows = score_dataset(manifests)
        for i, row in enumerate(rows[:10], 1):
            print(
                f"{i:3} {row['model'][:48]:48} {row['rate']:.3f} n={row['n']}"
                if row["rate"] != ""
                else f"{i:3} {row['model'][:48]:48}     -- no eligible span"
            )
        path = os.path.join(args.out_dir, f"ref_rendering_{dataset}.csv")
        write_csv(path, rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
