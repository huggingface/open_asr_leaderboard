#!/usr/bin/env python3
# Adapted from the reference implementation accompanying "Quantifying Benchmark
# Optimization in ASR Models" (https://github.com/tlebryk/asr-benchmark-optimization,
# Apache-2.0).
"""Score reference rendering agreement on the English short-form benchmarks.

For curated spelling, initialism, number, title, and compound variants erased by
English WER normalization, this asks whether a model reproduces the reference's
form after correctly transcribing the underlying words. One rate is written per
model and dataset.

References and hypotheses come from the published prediction manifests; no audio
or inference is required.

Usage:
    # Sync the public results bucket, then score every model in it.
    python benchmark_fitting/score_ref_rendering.py

    # Score an already-downloaded copy of the predictions.
    python benchmark_fitting/score_ref_rendering.py --preds_dir results

    # One dataset only.
    python benchmark_fitting/score_ref_rendering.py --preds_dir results --datasets voxpopuli_test

    # One model, printing its CSV line per dataset instead of writing files.
    python benchmark_fitting/score_ref_rendering.py --preds_dir results --model openai/whisper-large-v3

Writes `ref_rendering_<dataset>.csv` (one row per model) into `--out_dir`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

from ref_rendering_utils import REPORTED_CLASSES, score_pairs
from utils import BUCKET_RE, DEFAULT_BUCKET, read_manifest, resolve_model, sync_bucket
from utils import find_manifests as _find_manifests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from normalizer import to_hub_ids  # noqa: E402  (needs REPO_ROOT on sys.path)

# The English short-form sets, whose references the English normalizer applies
# to. Longest match wins when a manifest name is matched against these.
SHORT_FORM_DATASETS = (
    "ami_cleaned_test",
    "earnings22_test",
    "gigaspeech_cleaned_test",
    "librispeech_test.clean",
    "librispeech_test.other",
    "spgispeech_test",
    "voxpopuli_test",
    "voxpopuli_cleaned_aa_test",
)

# Manifest names in the results bucket, e.g.
# MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_test.jsonl


# ---------------------------------------------------------------------------
# Reading predictions
# ---------------------------------------------------------------------------




def _dataset_tag(name: str, datasets) -> str | None:
    """The dataset tag ``name`` ends with, longest first."""
    if "longform" in name.casefold():
        return None
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
    """Manifests for one short-form dataset, under ``root``.

    :func:`utils.find_manifests` with this scorer's dataset resolution: a name is
    matched against the known tags longest-first, so ``voxpopuli_cleaned_aa_test``
    is not read as a ``voxpopuli_test`` manifest, and longform runs are excluded.
    """
    return _find_manifests(
        root, dataset, dataset_of=lambda group: _dataset_tag(group, SHORT_FORM_DATASETS)
    )








# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_dataset(manifests: dict[str, str]) -> list[dict]:
    """One row per model, ordered by Hub id, case-insensitively.

    Rows are labelled with the Hub id the manifest was written from; the bucket
    name is what the files are keyed by, and is only a spelling of it. Ordering
    by name rather than by rate keeps a dataset's CSV diffable against the next
    run and against its sibling datasets, and matches the VoxPopuli scorer.
    """
    hub = to_hub_ids(sorted(manifests))
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
            out = {"model": hub[model], "rate": "", "n": 0}
        else:
            out = {"model": hub[model], "rate": scored[0] / scored[1], "n": scored[1]}
        for cls in REPORTED_CLASSES:
            k, n = agg["by_class"][cls]
            out[f"{cls}_rate"] = k / n if n else 0.0
            out[f"{cls}_n"] = n
        rows.append(out)
    # Exact id as a tiebreak, so the order stays deterministic if two Hub ids
    # ever differ only in case.
    rows.sort(key=lambda r: (r["model"].lower(), r["model"]))
    return rows


def by_rate(rows: list[dict]) -> list[dict]:
    """``rows`` ranked best-first, with the models that had no eligible span last."""
    return sorted(rows, key=lambda r: (r["rate"] == "", -(r["rate"] or 0.0)))


FIELDNAMES = ["model", "rate", "n"] + [
    f"{cls}_{suffix}" for cls in REPORTED_CLASSES for suffix in ("rate", "n")
]


def csv_row(row: dict) -> list:
    """One row in output order. Counts and the model id stay verbatim; a missing
    rate stays empty."""
    return [
        row[k] if k == "model" or k == "n" or k.endswith("_n") or row[k] == "" else f"{row[k]:.6f}"
        for k in FIELDNAMES
    ]


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        writer.writerows(csv_row(row) for row in rows)


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
    parser.add_argument(
        "--model",
        default=None,
        help="Score only this model and print its CSV line per dataset to stdout instead of "
        "writing the output files. Accepts the Hub id (openai/whisper-large-v3), the bucket id "
        "(openai-whisper-large-v3), or a unique substring of either.",
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

    by_dataset = {dataset: find_manifests(root, dataset) for dataset in datasets}

    selected = None
    if args.model:
        # Every model is scored independently against the reference, so the row
        # a single-model run prints is the row a full run would write for it.
        available: dict[str, str] = {}
        for manifests in by_dataset.values():
            available.update(manifests)
        selected = resolve_model(args.model, available)
        print(f"scoring only {selected}")

    if not selected:
        os.makedirs(args.out_dir, exist_ok=True)
    for dataset in datasets:
        manifests = by_dataset[dataset]
        print(f"\n=== {dataset}: {len(manifests)} manifests")
        if not manifests:
            continue
        if selected:
            if selected not in manifests:
                print(f"  no {selected} manifest for this dataset")
                continue
            manifests = {selected: manifests[selected]}
        rows = score_dataset(manifests)
        if selected:
            # A single-model run would otherwise overwrite each dataset's CSV
            # with a one-row file; print the row instead, ready to paste in.
            writer = csv.writer(sys.stdout)
            writer.writerow(FIELDNAMES)
            writer.writerows(csv_row(row) for row in rows)
            continue
        print("  top 10 by rate:")
        for i, row in enumerate(by_rate(rows)[:10], 1):
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
