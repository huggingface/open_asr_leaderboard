#!/usr/bin/env python3
# Adapted from the reference implementation accompanying "Quantifying Benchmark
# Optimization in ASR Models" (https://github.com/tlebryk/asr-benchmark-optimization,
# Apache-2.0).
"""Score reference error agreement rate on VoxPopuli.

VoxPopuli's official English references derive from parliamentary records rather
than from the audio, so some spans they contain were not spoken. The
`ArtificialAnalysis/VoxPopuli-Cleaned-AA` subset supplies a human-corrected
reference for 628 of those clips, and the leaderboard already reports WER against
it. Diffing the two references locates the spans where they disagree; for each
such span, a model's output either matches the official reference or matches the
correction.

Both reference sets are read out of the prediction manifests already published in
the results bucket: the `text` field of the `voxpopuli_test` manifests carries the
official reference, and the `text` field of the `voxpopuli_cleaned_aa_test`
manifests carries the correction. Hypotheses are read from the
`voxpopuli_cleaned_aa_test` manifests. No audio and no model inference required.

Usage:
    # Sync the public results bucket, then score every model in it.
    python benchmark_fitting/score_voxpopuli_ref_errors.py

    # Score an already-downloaded copy of the predictions.
    python benchmark_fitting/score_voxpopuli_ref_errors.py --preds_dir results

    # Score a single model and print its CSV line instead of writing files.
    python benchmark_fitting/score_voxpopuli_ref_errors.py --preds_dir results --model openai/whisper-large-v3

Each model is also scored for WER against both references over the same clips, so
the difference between the two is the WER penalty the official reference's errors
impose on it.

Outputs `ref_error_agreement_voxpopuli.csv` (one row per model) and
`edits_voxpopuli.jsonl` (one row per disagreement) in `--out_dir`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

from kaldialign import batch_error_rate

from ref_error_utils import DEFAULTS, find_ref_errors, ref_error_agreement
from utils import DEFAULT_BUCKET, find_manifests, read_manifest, resolve_model, sync_bucket

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# Allow importing the repo's normalizer regardless of where this is called from.
sys.path.insert(0, REPO_ROOT)

from normalizer import to_hub_ids  # noqa: E402  (needs REPO_ROOT on sys.path)

OFFICIAL_DATASET = "voxpopuli_test"
CORRECTED_DATASET = "voxpopuli_cleaned_aa_test"

# Manifests written without audio file paths key their rows `sample_0`,
# `sample_1`, ... . Those cannot be joined on the key, only on row order.
OPAQUE_KEY_PREFIX = "sample_"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _make_normalizer():
    """This repo's English normalizer, as instantiated by ``normalizer.data_utils``.

    Reference and hypothesis are normalized identically, so a span only counts as
    a disagreement when it survives the same text normalization the leaderboard's
    WER is computed under. Not interchangeable with the upstream Whisper
    normalizer, which lacks the acronym, name and compound stages.
    """
    from normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


_NORMALIZER = None


def normalize(text: str) -> str:
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = _make_normalizer()
    if not text:
        return ""
    return re.sub(r"\s+", " ", _NORMALIZER(text)).strip()


# ---------------------------------------------------------------------------
# Reading predictions
# ---------------------------------------------------------------------------


REQUIRED_FIELDS = ("audio_filepath", "text", "pred_text")






def collect_references(manifests: dict[str, str], dataset: str) -> tuple[list[str], dict[str, str]]:
    """Read the reference transcripts a dataset's manifests all share.

    Returns every clip key seen across the key-bearing manifests, in first-seen
    order and deduplicated, plus a key-to-text map. Manifests for a dataset
    normally carry identical references; where they disagree the majority text
    wins and the disagreement is reported, so that one manifest built against a
    different dataset revision cannot abort or silently shrink the run.
    """
    keys_in_order: list[str] = []
    seen: set[str] = set()
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in manifests.values():
        rows = read_manifest(path, REQUIRED_FIELDS)
        keys = [row["audio_filepath"] for row in rows]
        if any(key.startswith(OPAQUE_KEY_PREFIX) for key in keys):
            continue
        for row in rows:
            key = row["audio_filepath"]
            if key not in seen:
                seen.add(key)
                keys_in_order.append(key)
            votes[key][row["text"]] += 1
    if not keys_in_order:
        raise ValueError(f"no key-bearing manifest found for {dataset}")
    text_by_key = {key: counter.most_common(1)[0][0] for key, counter in votes.items()}
    contested = sum(1 for counter in votes.values() if len(counter) > 1)
    if contested:
        print(f"  note: {contested}/{len(keys_in_order)} clips have manifests disagreeing on the reference; using the majority")
    return keys_in_order, text_by_key


def collect_hypotheses(
    manifests: dict[str, str], keys_in_order: list[str]
) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]]:
    """Map model name to its key-to-hypothesis map.

    Manifests keyed ``sample_<i>`` are joined on row order, and only when their
    reference column matches the key-bearing manifests row for row; otherwise the
    model is skipped, since a wrong join would silently mis-score it.
    """
    anchor_text: list[str | None] = [None] * len(keys_in_order)
    hypotheses: dict[str, dict[str, str]] = {}
    skipped: list[tuple[str, str]] = []

    for model, path in manifests.items():
        rows = read_manifest(path, REQUIRED_FIELDS)
        keys = [row["audio_filepath"] for row in rows]
        if not any(key.startswith(OPAQUE_KEY_PREFIX) for key in keys):
            hypotheses[model] = {row["audio_filepath"]: row["pred_text"] for row in rows}
            if keys_in_order and anchor_text[0] is None and keys == keys_in_order:
                anchor_text = [row["text"] for row in rows]

    for model, path in manifests.items():
        if model in hypotheses:
            continue
        rows = read_manifest(path, REQUIRED_FIELDS)
        if len(rows) != len(keys_in_order):
            skipped.append((model, f"{len(rows)} rows, expected {len(keys_in_order)}"))
            continue
        mismatches = sum(1 for i, row in enumerate(rows) if row["text"] != anchor_text[i])
        if mismatches:
            skipped.append((model, f"{mismatches}/{len(rows)} references disagree by row order"))
            continue
        hypotheses[model] = {keys_in_order[i]: rows[i]["pred_text"] for i in range(len(rows))}

    return hypotheses, skipped


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------


def corpus_wer(pairs: list[tuple[str, str]]) -> float | None:
    """Corpus WER over already-normalized ``(reference, hypothesis)`` pairs.

    ``kaldialign.batch_error_rate`` with ``merge_compounds=True``: the same call
    ``api/run_eval.py`` makes, so these numbers sit on the leaderboard's scale
    rather than on a second, subtly different one. Corpus-level — total errors
    over total reference words — not a mean of per-clip rates.
    """
    if not pairs:
        return None
    refs = [tuple(ref.split()) for ref, _ in pairs]
    hyps = [tuple(hyp.split()) for _, hyp in pairs]
    return 100 * batch_error_rate(refs, hyps, merge_compounds=True)["err_rate"]


def score_wer(
    keys: list[str],
    official: dict[str, str],
    corrected: dict[str, str],
    hypotheses: dict[str, dict[str, str]],
) -> dict[str, dict]:
    """Each model's WER against both references, over one shared clip set.

    A model is scored on the clips it has a hypothesis for and both references
    cover, and both WERs are computed over exactly that set. The two therefore
    differ only in which reference they score against, so their difference is
    the cost the official reference's errors impose on that model.
    """
    out: dict[str, dict] = {}
    for model, by_key in hypotheses.items():
        official_pairs: list[tuple[str, str]] = []
        corrected_pairs: list[tuple[str, str]] = []
        for key in keys:
            hyp = by_key.get(key)
            # An empty reference contributes no words to the denominator but
            # would still charge insertions; drop it from both sides alike.
            if hyp is None or not official[key] or not corrected[key]:
                continue
            official_pairs.append((official[key], hyp))
            corrected_pairs.append((corrected[key], hyp))
        out[model] = {
            "wer_official": corpus_wer(official_pairs),
            "wer_corrected": corpus_wer(corrected_pairs),
            "n_wer_clips": len(official_pairs),
        }
    return out








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
    parser.add_argument("--official_dataset", default=OFFICIAL_DATASET)
    parser.add_argument("--corrected_dataset", default=CORRECTED_DATASET)
    parser.add_argument(
        "--model",
        default=None,
        help="Score only this model and print its CSV line to stdout instead of writing the "
        "output files. Accepts the Hub id (openai/whisper-large-v3), the bucket id "
        "(openai-whisper-large-v3), or a unique substring of either.",
    )
    args = parser.parse_args()

    if args.preds_dir:
        root = args.preds_dir
        print(f"Reading predictions from {root}\n")
    else:
        root = args.local_dir
        sync_bucket(args.bucket, root, hf_token=args.hf_token)

    official_manifests = find_manifests(root, args.official_dataset)
    corrected_manifests = find_manifests(root, args.corrected_dataset)
    print(f"{args.official_dataset}: {len(official_manifests)} manifests")
    print(f"{args.corrected_dataset}: {len(corrected_manifests)} manifests")
    if not official_manifests or not corrected_manifests:
        sys.exit("Both datasets are required; nothing to score.")

    _, official_ref = collect_references(official_manifests, args.official_dataset)
    keys_in_order, corrected_ref = collect_references(corrected_manifests, args.corrected_dataset)
    keys = [key for key in keys_in_order if key in official_ref]
    print(f"clips with both references: {len(keys)} / {len(keys_in_order)}")

    hypotheses, skipped = collect_hypotheses(corrected_manifests, keys_in_order)
    print(f"models with usable hypotheses: {len(hypotheses)} / {len(corrected_manifests)}")
    for model, why in skipped:
        print(f"  skipped {model}: {why}")

    if args.model:
        # Edits come from diffing the two references, and each verdict is
        # decided from that model's hypothesis alone, so restricting the set
        # here yields exactly the row a full run would produce for it.
        selected = resolve_model(args.model, hypotheses, skipped, what="scored model")
        hypotheses = {selected: hypotheses[selected]}
        print(f"scoring only {selected}")

    # Normalize once up front: the edit finder and the WER pass below must see
    # the same text, and re-normalizing per pass would double the cost of the run.
    norm_official = {key: normalize(official_ref[key]) for key in keys}
    norm_corrected = {key: normalize(corrected_ref[key]) for key in keys}
    norm_hyps = {
        model: {key: normalize(by_key[key]) for key in keys if key in by_key}
        for model, by_key in hypotheses.items()
    }

    edits = []
    for key in keys:
        ref_tokens = norm_official[key].split()
        if not ref_tokens:
            continue
        model_hyps = {model: by_key[key] for model, by_key in norm_hyps.items() if key in by_key}
        if not model_hyps:
            continue
        clip_edits = find_ref_errors(
            ref_tokens,
            norm_corrected[key],
            model_hyps,
            min_run_len=DEFAULTS["min_run_len"],
            include_middle=DEFAULTS["include_middle"],
            min_ref_match=DEFAULTS["min_ref_match"],
            min_span_distance=DEFAULTS["min_span_distance"],
        )
        for edit in clip_edits:
            edit.clip_key = key
        edits.extend(clip_edits)

    by_kind: dict[str, int] = collections.defaultdict(int)
    by_position: dict[str, int] = collections.defaultdict(int)
    for edit in edits:
        by_kind[edit.kind] += 1
        by_position[edit.position] += 1
    print(f"\nreference errors found: {len(edits)}")
    print(f"  by kind: {dict(sorted(by_kind.items()))}")
    print(f"  by position: {dict(sorted(by_position.items()))}")

    rates = ref_error_agreement(edits)
    print(f"models scored: {len(rates)}")

    wers = score_wer(keys, norm_official, norm_corrected, norm_hyps)

    header = [
        "model",
        "rate",
        "n_ref",
        "n_eligible",
        "n_clips",
        "wer_official",
        "wer_corrected",
        "n_wer_clips",
    ]
    hub = to_hub_ids(sorted(rates))
    rows = []
    # Case-insensitive, so `nvidia/...` and `OpenMOSS-Team/...` interleave by name
    # rather than splitting into an upper-case block and a lower-case one.
    for model in sorted(rates, key=lambda m: (hub[m].lower(), hub[m])):
        wer = wers.get(model, {})
        rows.append(
            [
                hub[model],
                f"{rates[model]['rate']:.4f}",
                rates[model]["n_ref"],
                rates[model]["n_eligible"],
                rates[model]["n_clips"],
                # Blank rather than 0 when a model has no scorable clip: an
                # absent WER is not a perfect one.
                "" if wer.get("wer_official") is None else f"{wer['wer_official']:.2f}",
                "" if wer.get("wer_corrected") is None else f"{wer['wer_corrected']:.2f}",
                wer.get("n_wer_clips", 0),
            ]
        )

    if args.model:
        # A single-model run would otherwise overwrite a full run's outputs with
        # a one-row file; print the row instead, ready to paste into the CSV.
        print()
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(rows)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "ref_error_agreement_voxpopuli.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    edits_path = os.path.join(args.out_dir, "edits_voxpopuli.jsonl")
    with open(edits_path, "w", encoding="utf-8") as f:
        for edit in edits:
            f.write(
                json.dumps(
                    {
                        "clip_key": edit.clip_key,
                        "kind": edit.kind,
                        "position": edit.position,
                        # An "insert" edit is a span the official reference omits,
                        # so its official side is empty by construction.
                        "official_span": edit.text if edit.kind == "delete" else "",
                        "corrected_span": edit.corrected_text,
                        "ref_indices": edit.ref_indices,
                        "span_distance": edit.consensus_distance,
                        # Keyed by Hub id, to match the CSV's model column.
                        "verdict": {hub.get(m, m): v for m, v in edit.verdict.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {edits_path}")


if __name__ == "__main__":
    main()
