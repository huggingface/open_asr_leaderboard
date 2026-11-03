#!/usr/bin/env python3
"""Rescore every published row after a normalizer change.

Nothing is pushed unless --open_pr is passed: the default is a dry run that prints the
changes. With --open_pr the recomputed `.score.json` files are also uploaded to the
buckets, so later `open_results_pr.py` runs read scores from the new normalizer.

Usage:
    # dry run over every row of every results file
    python scripts/rescore_all.py

    # some models / files only, keeping the updated CSVs locally for inspection
    python scripts/rescore_all.py --models openai/whisper-large-v3 nvidia/canary-1b-v2 \\
        --targets english multilingual --language de --out_dir /tmp/rescored

    # re-use results already synced locally, and open the PRs
    python scripts/rescore_all.py --skip_sync --open_pr
"""

import argparse
import multiprocessing
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

from open_results_pr import (  # noqa: E402
    ALL_TARGETS,
    METADATA_ARGS,
    PRIVATE_BUCKET,
    build_targets,
    collect_row,
    fetch_csv,
    recompute_averages,
    write_csv,
)
from normalizer.model_ids import BUCKET_ID_OVERRIDES  # noqa: E402
from score_bucket_results import ML_LANGUAGES, sync_bucket, upload_scores  # noqa: E402

# Published row name -> model id the results bucket uses, for rows named otherwise.
ROW_TO_BUCKET_ID = {hub_id: bucket_id for bucket_id, hub_id in BUCKET_ID_OVERRIDES.items()}


def as_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def is_rtfx(column):
    return column == "RTFx" or column.endswith(" RTFx")


def rtfx_column(header, column):
    """The published RTFx column measuring the same manifests as `column`, if any.

    The file-wide RTFx is used only in files without per-dataset RTFx (multilingual,
    long-form): on the English sheet it covers a different set of datasets.
    """
    if column.endswith(" WER") and f"{column[:-len(' WER')]} RTFx" in header:
        return f"{column[:-len(' WER')]} RTFx"
    if "RTFx" in header and not any(c.endswith(" RTFx") for c in header):
        return "RTFx"
    return None


def rescore_row(target, header, old_row, values, rtfx_tol, force):
    """New row with only the WER cells (and averages) replaced. Returns (row, changes, skipped)."""
    columns = {column: i for i, column in enumerate(header)}
    padded = old_row + [""] * (len(header) - len(old_row))
    new_row = list(padded)
    skipped = []
    for column, value in values.items():
        if (value == "" or column not in columns or is_rtfx(column) or column in target.ignore
                or column in METADATA_ARGS):
            continue
        if not padded[columns[column]].strip():
            continue  # never published (e.g. left out on purpose): rescoring must not add it
        rtfx = rtfx_column(header, column)
        old_rtfx = as_float(padded[columns[rtfx]]) if rtfx else None
        new_rtfx = as_float(values.get(rtfx))
        # API rows publish RTFx as blank or -1, so there is nothing to check against.
        if (not force and old_rtfx is not None and old_rtfx > 0 and new_rtfx is not None
                and abs(old_rtfx - new_rtfx) > rtfx_tol):
            skipped.append(f"{column} ({rtfx} {old_rtfx:g} published vs {new_rtfx:g} rescored)")
            continue
        new_row[columns[column]] = value
    recompute_averages(target, header, new_row)

    changes = [
        (column, old, new)
        for column, old, new in zip(header, padded, new_row)
        if old != new and as_float(old) != as_float(new)
    ]
    return new_row, changes, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Rescore every published leaderboard row after a normalizer change.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=ALL_TARGETS,
        choices=ALL_TARGETS,
        help="Results files to rescore (default: all).",
    )
    parser.add_argument(
        "--language",
        nargs="+",
        default=ML_LANGUAGES,
        choices=ML_LANGUAGES,
        help="Languages of the multilingual target to rescore (default: all).",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Only these published row names.")
    parser.add_argument("--local_dir", default=None, help="Where to sync results (default: <repo>/results).")
    parser.add_argument("--skip_sync", action="store_true", help="Score already-downloaded results.")
    parser.add_argument(
        "--rtfx_tol",
        type=float,
        default=0.01,
        help="Max |published - rescored| RTFx before a column is treated as scored from other "
             "manifests and skipped (default: 0.01, the published rounding).",
    )
    parser.add_argument("--force", action="store_true", help="Update WER even when RTFx does not match.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Processes used to align the transcripts of each manifest (default: every CPU).",
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN.")
    parser.add_argument("--out_dir", default=None, help="Also write the updated CSVs to this directory.")
    parser.add_argument("--open_pr", action="store_true", help="Actually open the PRs (default: dry run).")
    parser.add_argument("--commit_message", default="Rescore WER after normalizer change", help="PR title.")
    args = parser.parse_args()

    # Scoring aligns transcripts in a process pool. Forking after huggingface_hub has
    # started its download threads can leave the workers deadlocked on a copied lock.
    multiprocessing.set_start_method("forkserver")

    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")
    targets = [
        build_targets(language)[key]
        for key in args.targets
        for language in (args.language if key == "multilingual" else [None])
    ]
    published = {target.filename: fetch_csv(target.repo_id, target.filename, args.hf_token)
                 for target in targets}

    # Row name -> the targets that publish it.
    jobs = {}
    for target in targets:
        for row in published[target.filename][1]:
            name = row[0].strip()
            if not args.models or name in args.models:
                jobs.setdefault(name, []).append(target)

    report, not_rescored, written_scores, buckets = [], [], [], set()
    for name, row_targets in sorted(jobs.items()):
        if " (" in name:
            not_rescored.append(f"{name}: variant row, its manifests cannot be told apart")
            continue
        model_id = ROW_TO_BUCKET_ID.get(name, name)
        print(f"\n=== {name}")
        # API models land in the private bucket whatever sheet they are published on.
        model_buckets = {target.bucket for target in row_targets} | {PRIVATE_BUCKET}
        buckets |= model_buckets
        if not args.skip_sync:
            try:
                for bucket in sorted(model_buckets):
                    sync_bucket(bucket, local_dir, hf_token=args.hf_token, model_ids=[model_id])
            except subprocess.CalledProcessError as exc:
                not_rescored.append(f"{name}: sync failed (exit {exc.returncode})")
                continue

        for target in row_targets:
            values = collect_row(target, local_dir, model_id, recompute=True,
                                 written_scores=written_scores, num_workers=args.num_workers)
            if not values:
                not_rescored.append(f"{name}: no results for {target.filename}")
                continue
            header, rows = published[target.filename]
            for index, row in enumerate(rows):
                if row[0].strip() != name:
                    continue
                rows[index], changes, skipped = rescore_row(
                    target, header, row, values, args.rtfx_tol, args.force
                )
                report.append((target.filename, name, changes, skipped))

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80 + "\nChanges\n" + "=" * 80)
    changed_files = set()
    for filename, name, changes, skipped in report:
        if not changes and not skipped:
            continue
        print(f"\n{filename} | {name}")
        for column, old, new in changes:
            print(f"  {column:<36} {old or '(blank)':>8} -> {new or '(blank)'}")
        for note in skipped:
            print(f"  SKIPPED {note}")
        if changes:
            changed_files.add(filename)
    unchanged = sum(1 for _, _, changes, skipped in report if not changes and not skipped)
    print(f"\n{unchanged} row(s) unchanged.")
    if not_rescored:
        print("\nNot rescored:")
        for note in not_rescored:
            print(f"  {note}")

    updated = {}  # repo id -> {filename: content}
    for target in targets:
        if target.filename in changed_files:
            updated.setdefault(target.repo_id, {})[target.filename] = write_csv(*published[target.filename])
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for files in updated.values():
            for filename, content in files.items():
                path = os.path.join(args.out_dir, filename)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                print(f"Wrote {path}")

    if not updated:
        print("\nNothing to submit.")
        return
    if not args.open_pr:
        print(f"\nDry run - not opening a PR. {len(changed_files)} file(s) would change: "
              f"{', '.join(sorted(changed_files))}.\nRe-run with --open_pr to submit.")
        return

    for bucket in sorted(buckets):
        upload_scores(bucket, local_dir, written_scores, hf_token=args.hf_token)
    api = HfApi(token=args.hf_token)
    for repo_id, files in sorted(updated.items()):
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=filename, path_or_fileobj=content.encode("utf-8"))
                for filename, content in sorted(files.items())
            ],
            commit_message=args.commit_message,
            commit_description=(
                f"WER rescored with `scripts/rescore_all.py`.\n\nFiles: {', '.join(sorted(files))}"
            ),
            create_pr=True,
        )
        print(f"\nPR opened on {repo_id}: {commit.pr_url}")


if __name__ == "__main__":
    main()
