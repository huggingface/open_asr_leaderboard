"""Aggregate per-shard checkpoints into a leaderboard-style summary.

Reads every ``CHK_*.json`` snapshot written by run_eval.py under a given
results directory, concatenates per-dataset predictions/references, computes
WER per dataset with evaluate.load('wer'), and prints the macro-average WER
across all 8 datasets so we can gauge progress against the 5.30% target
without waiting for the run to finish.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import evaluate


wer_metric = evaluate.load("wer")


DATASETS = {
    "ami": ["test"],
    "earnings22": ["test"],
    "gigaspeech": ["test"],
    "librispeech": ["test.clean", "test.other"],
    "spgispeech": ["test"],
    "tedlium": ["test"],
    "voxpopuli": ["test"],
}


def parse_chk_name(fn):
    # CHK_<model>_<dataset>_<split>_shard<i>-<n>.json
    name = os.path.basename(fn)[:-5]
    if not name.startswith("CHK_"):
        return None
    parts = name[4:].rsplit("_shard", 1)
    head = parts[0]
    shard_part = parts[1]
    # head = <model>_<dataset>_<split>
    # pop the split off the right; split may have a dot (test.clean)
    toks = head.split("_")
    # model contains dashes, dataset is single token (ami, earnings22, ...)
    # try matching known datasets from right
    for ds in DATASETS:
        for split in DATASETS[ds]:
            suffix = f"{ds}_{split}"
            if head.endswith(suffix):
                model = head[: -len(suffix) - 1]
                return model, ds, split, shard_part
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    chks = sorted(glob.glob(os.path.join(args.dir, "CHK_*.json")))
    if not chks:
        print(f"No CHK_*.json under {args.dir}", file=sys.stderr)
        sys.exit(1)

    # (model, dataset, split) -> merged {predictions, references, audio_len, time}
    buckets = defaultdict(lambda: {"predictions": [], "references": [],
                                    "audio_length_s": [], "transcription_time_s": []})
    for chk in chks:
        meta = parse_chk_name(chk)
        if not meta:
            print("skip", chk); continue
        model, ds, split, _ = meta
        with open(chk) as f:
            d = json.load(f)
        key = (model, ds, split)
        for k in buckets[key]:
            buckets[key][k].extend(d.get(k, []))

    # Per-dataset WER (collapse librispeech clean/other into two entries, avg'd).
    print(f"{'dataset':<24}{'n':>8}  {'WER%':>7}  {'audio_h':>8}  {'RTFx':>7}")
    print("-" * 60)
    per_ds = {}
    for (model, ds, split), v in sorted(buckets.items()):
        n = len(v["predictions"])
        if n == 0:
            continue
        try:
            wer = wer_metric.compute(
                references=v["references"], predictions=v["predictions"]
            )
            wer_pct = round(100 * wer, 2)
        except Exception as e:
            wer_pct = float("nan")
        aud_h = sum(v["audio_length_s"]) / 3600.0
        tot_t = sum(v["transcription_time_s"]) or 1.0
        rtfx = sum(v["audio_length_s"]) / tot_t
        label = f"{ds}/{split}"
        print(f"{label:<24}{n:>8}  {wer_pct:>7.2f}  {aud_h:>8.2f}  {rtfx:>7.2f}")
        per_ds[(ds, split)] = wer_pct

    # Macro-average across 8 leaderboard datasets (ami, e22, gs, ls.c, ls.o, spg, ted, vp)
    lb8 = [
        ("ami", "test"),
        ("earnings22", "test"),
        ("gigaspeech", "test"),
        ("librispeech", "test.clean"),
        ("librispeech", "test.other"),
        ("spgispeech", "test"),
        ("tedlium", "test"),
        ("voxpopuli", "test"),
    ]
    vals = [per_ds.get(k) for k in lb8]
    have = [v for v in vals if v is not None]
    if len(have) == 8:
        avg = round(sum(have) / 8, 2)
        print(f"\nMacro avg WER over 8 datasets: {avg}%")
    else:
        present = [f"{k[0]}/{k[1]}" for k, v in zip(lb8, vals) if v is not None]
        missing = [f"{k[0]}/{k[1]}" for k, v in zip(lb8, vals) if v is None]
        print(f"\nPartial: {len(have)}/8 datasets scored.  present={present}  missing={missing}")


if __name__ == "__main__":
    main()
