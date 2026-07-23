#!/usr/bin/env python3
"""Score a complete long-form run and compare it with the leaderboard CSV."""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jiwer import wer
from normalizer import EnglishTextNormalizer


LONGFORM_DATASETS = {
    "earnings21": "asr-leaderboard-longform_earnings21_test",
    "earnings22": "asr-leaderboard-longform_earnings22_test",
    "tedlium": "distil-whisper-tedlium-long-form_default_test",
}
CORAAL_SUBSETS = ("ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD")


def find_result(results, suffix):
    matches = [value for key, value in results.items() if suffix in key]
    if len(matches) != 1:
        raise ValueError(f"Expected one result matching {suffix!r}, found {len(matches)}")
    return matches[0]


def load_current_row(path, model_id):
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model_id"] == model_id:
                return row
    return None


def format_delta(new, old, digits=2):
    delta = new - old
    return f"{delta:+.{digits}f}"


def score_manifests(results_dir, model_id):
    normalizer = EnglishTextNormalizer()
    model_slug = model_id.replace("/", "-")
    results = {}
    pattern = str(results_dir / f"**/MODEL_{model_slug}_DATASET_*.jsonl")
    for manifest_path in sorted(glob.glob(pattern, recursive=True)):
        references = []
        predictions = []
        durations = []
        times = []
        with open(manifest_path, encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                references.append(normalizer(sample["text"]))
                predictions.append(normalizer(sample["pred_text"]))
                durations.append(sample["duration"])
                times.append(sample["time"])

        dataset_id = Path(manifest_path).stem.split("_DATASET_", 1)[1]
        results[dataset_id] = {
            "wer": round(100 * wer(references, predictions), 2),
            "audio_length": sum(durations),
            "inference_time": sum(times),
        }
    if not results:
        raise ValueError(f"No result files found for {model_id} in {results_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--current-csv", type=Path)
    args = parser.parse_args()

    raw_results = score_manifests(args.results_dir, args.model_id)

    metrics = {
        name: find_result(raw_results, suffix)
        for name, suffix in LONGFORM_DATASETS.items()
    }
    coraal_metrics = [
        find_result(raw_results, f"bezzam-coraal_{subset}_test")
        for subset in CORAAL_SUBSETS
    ]
    coraal_wer = sum(metric["wer"] for metric in coraal_metrics) / len(coraal_metrics)
    without_coraal = sum(metric["wer"] for metric in metrics.values()) / len(metrics)
    average = (sum(metric["wer"] for metric in metrics.values()) + coraal_wer) / 4

    all_metrics = [*metrics.values(), *coraal_metrics]
    total_audio = sum(metric["audio_length"] for metric in all_metrics)
    total_time = sum(metric["inference_time"] for metric in all_metrics)
    rtfx = total_audio / total_time

    print("\nLong-form leaderboard row:")
    print("model_id,Average,RTFx,earnings21,earnings22,tedlium,coraal_avg,Avg (without CORAAL)")
    print(
        f"{args.model_id},{average:.4f},{rtfx:.2f},"
        f"{metrics['earnings21']['wer']:.2f},{metrics['earnings22']['wer']:.2f},"
        f"{metrics['tedlium']['wer']:.2f},{coraal_wer:.4f},{without_coraal:.4f}"
    )

    current = load_current_row(args.current_csv, args.model_id)
    if current is None:
        return

    old_average = float(current["Average"])
    old_rtfx = float(current["RTFx"])
    print("\nComparison with current leaderboard:")
    print("| Metric | Current | H200 | Change |")
    print("| --- | ---: | ---: | ---: |")
    if current["coraal_avg"].strip():
        print(f"| Average WER | {old_average:.2f} | {average:.2f} | {format_delta(average, old_average)} |")
    else:
        print(
            f"| Three-split average WER | {old_average:.2f} | {without_coraal:.2f} | "
            f"{format_delta(without_coraal, old_average)} |"
        )
        print(f"| Full average WER (now including CORAAL) | n/a | {average:.2f} | new |")
    print(f"| RTFx | {old_rtfx:.2f} | {rtfx:.2f} | {rtfx / old_rtfx:.2f}x |")
    for key, label in (("earnings21", "Earnings21 WER"), ("earnings22", "Earnings22 WER"), ("tedlium", "TED-LIUM WER")):
        old = float(current[key])
        new = metrics[key]["wer"]
        print(f"| {label} | {old:.2f} | {new:.2f} | {format_delta(new, old)} |")
    if current["coraal_avg"].strip():
        old_coraal = float(current["coraal_avg"])
        print(f"| CORAAL WER | {old_coraal:.2f} | {coraal_wer:.2f} | {format_delta(coraal_wer, old_coraal)} |")
    else:
        print(f"| CORAAL WER | n/a | {coraal_wer:.2f} | new |")


if __name__ == "__main__":
    main()
