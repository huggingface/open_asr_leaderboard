"""Validate multilingual ONNX artifacts and update the leaderboard CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CONFIG_TO_COLUMN = {
    "mcv_de": "de_covost",
    "fleurs_de": "de_fleurs",
    "mcv_fr": "fr_covost",
    "mls_fr": "fr_mls",
    "fleurs_fr": "fr_fleurs",
    "mcv_it": "it_covost",
    "mls_it": "it_mls",
    "fleurs_it": "it_fleurs",
    "mcv_es": "es_covost",
    "mls_es": "es_mls",
    "fleurs_es": "es_fleurs",
    "mls_pt": "pt_mls",
    "fleurs_pt": "pt_fleurs",
}

CSV_FIELDS = [
    "model",
    "Model size (B)",
    "RTFx",
    "de_covost",
    "de_fleurs",
    "fr_covost",
    "fr_mls",
    "fr_fleurs",
    "it_covost",
    "it_mls",
    "it_fleurs",
    "es_covost",
    "es_mls",
    "es_fleurs",
    "pt_mls",
    "pt_fleurs",
    "Avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=MODEL_ID,SIZE_B",
        help="Expected variant specification; may be supplied more than once.",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def parse_variant_specs(raw_specs: list[str]) -> dict[str, tuple[str, str]]:
    specs: dict[str, tuple[str, str]] = {}
    for raw in raw_specs:
        try:
            name, remainder = raw.split("=", 1)
            model_id, size = remainder.rsplit(",", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --variant value: {raw!r}") from exc
        if not name or not model_id or not size or name in specs:
            raise ValueError(f"Invalid or duplicate --variant value: {raw!r}")
        specs[name] = (model_id, size)
    return specs


def display_number(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def load_metrics(artifact_root: Path) -> list[dict]:
    paths = sorted(artifact_root.glob("**/metrics.json"))
    if not paths:
        raise ValueError(f"No metrics.json files found below {artifact_root}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def build_rows(metrics: list[dict], specs: dict[str, tuple[str, str]]) -> tuple[list[dict[str, str]], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in metrics:
        grouped[result["variant"]].append(result)

    if set(grouped) != set(specs):
        raise ValueError(f"Expected variants {sorted(specs)}, found {sorted(grouped)}")

    rows: list[dict[str, str]] = []
    summary_variants: dict[str, dict] = {}
    for variant, (model_id, size) in specs.items():
        results = grouped[variant]
        by_config = {result["config_name"]: result for result in results}
        if len(by_config) != len(results):
            raise ValueError(f"Duplicate config results for {variant}")
        if set(by_config) != set(CONFIG_TO_COLUMN):
            missing = sorted(set(CONFIG_TO_COLUMN) - set(by_config))
            extra = sorted(set(by_config) - set(CONFIG_TO_COLUMN))
            raise ValueError(f"Invalid config set for {variant}; missing={missing}, extra={extra}")
        for result in results:
            if result["model_id"] != model_id:
                raise ValueError(
                    f"Model mismatch for {variant}/{result['config_name']}: "
                    f"{result['model_id']!r} != {model_id!r}"
                )
            if result["processed_samples"] <= 0 or result["total_inference_seconds"] <= 0:
                raise ValueError(f"Empty or invalid result for {variant}/{result['config_name']}")

        total_audio = sum(float(result["total_audio_seconds"]) for result in results)
        total_time = sum(float(result["total_inference_seconds"]) for result in results)
        wers = [float(by_config[config]["wer"]) for config in CONFIG_TO_COLUMN]
        row = {field: "" for field in CSV_FIELDS}
        row["model"] = model_id
        row["Model size (B)"] = size
        row["RTFx"] = display_number(total_audio / total_time)
        for config, column in CONFIG_TO_COLUMN.items():
            row[column] = display_number(float(by_config[config]["wer"]))
        row["Avg"] = display_number(sum(wers) / len(wers), places=9)
        rows.append(row)

        summary_variants[variant] = {
            "model_id": model_id,
            "model_size_b": float(size),
            "rtfx": total_audio / total_time,
            "average_wer": sum(wers) / len(wers),
            "total_audio_seconds": total_audio,
            "total_inference_seconds": total_time,
            "processed_samples": sum(int(result["processed_samples"]) for result in results),
            "configs": {config: by_config[config] for config in sorted(by_config)},
        }
    return rows, {"schema_version": 1, "variants": summary_variants}


def update_csv(path: Path, new_rows: list[dict[str, str]]) -> None:
    line_ending = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Unexpected CSV fields: {reader.fieldnames}")
        existing = list(reader)

    new_models = {row["model"] for row in new_rows}
    combined = [row for row in existing if row["model"] not in new_models] + new_rows
    combined.sort(key=lambda row: float(row["Avg"]) if row.get("Avg") else float("inf"))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator=line_ending)
        writer.writeheader()
        writer.writerows(combined)


def main() -> None:
    args = parse_args()
    specs = parse_variant_specs(args.variant)
    rows, summary = build_rows(load_metrics(args.artifact_root), specs)
    if args.csv:
        update_csv(args.csv, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
