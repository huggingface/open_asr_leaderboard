"""Shared specification and tooling for multilingual API benchmark runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_PATH = "nithinraok/asr-leaderboard-datasets"
SPLIT = "test"


@dataclass(frozen=True)
class BenchmarkConfig:
    config_name: str
    column: str
    language: str


# Keep this in leaderboard column order. The Mozilla Common Voice source splits
# are intentionally published under the historical CoVoST column names.
SCORE_CONFIGS = (
    BenchmarkConfig("mcv_de", "de_covost", "de"),
    BenchmarkConfig("fleurs_de", "de_fleurs", "de"),
    BenchmarkConfig("mcv_fr", "fr_covost", "fr"),
    BenchmarkConfig("mls_fr", "fr_mls", "fr"),
    BenchmarkConfig("fleurs_fr", "fr_fleurs", "fr"),
    BenchmarkConfig("mcv_it", "it_covost", "it"),
    BenchmarkConfig("mls_it", "it_mls", "it"),
    BenchmarkConfig("fleurs_it", "it_fleurs", "it"),
    BenchmarkConfig("mcv_es", "es_covost", "es"),
    BenchmarkConfig("mls_es", "es_mls", "es"),
    BenchmarkConfig("fleurs_es", "es_fleurs", "es"),
    BenchmarkConfig("mls_pt", "pt_mls", "pt"),
    BenchmarkConfig("fleurs_pt", "pt_fleurs", "pt"),
)

# Small FLEURS splits run first so credential/configuration mistakes surface
# before the more expensive Common Voice and MLS jobs.
EXECUTION_ORDER = (
    "fleurs_de",
    "fleurs_fr",
    "fleurs_it",
    "fleurs_es",
    "fleurs_pt",
    "mcv_de",
    "mcv_es",
    "mcv_fr",
    "mcv_it",
    "mls_es",
    "mls_fr",
    "mls_it",
    "mls_pt",
)

_MODEL_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def validate_model_id(model_id: str) -> str:
    parts = model_id.split("/")
    if len(parts) != 2 or not all(_MODEL_PART.fullmatch(part) for part in parts):
        raise ValueError(
            "model_id must be '<registered-provider>/<model-variant>' using only "
            "letters, numbers, '.', '_', ':', '@', '+', or '-'"
        )
    return model_id


def validate_env_name(value: str, label: str) -> str:
    if not _ENV_NAME.fullmatch(value):
        raise ValueError(f"{label} must be an uppercase environment-variable name")
    return value


def validate_run_id(value: str, label: str) -> str:
    if value and (not value.isdigit() or int(value) < 1):
        raise ValueError(f"{label} must be empty or a positive GitHub Actions run ID")
    return value


def validate_range(value: int, label: str, minimum: int, maximum: int) -> int:
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def model_slug(model_id: str) -> str:
    return validate_model_id(model_id).replace("/", "-")


def matrix_payload() -> dict[str, list[dict[str, str]]]:
    by_name = {item.config_name: item for item in SCORE_CONFIGS}
    if set(by_name) != set(EXECUTION_ORDER):
        raise RuntimeError("Scoring and execution benchmark configurations differ")
    return {
        "include": [
            {
                "config": by_name[name].config_name,
                "language": by_name[name].language,
            }
            for name in EXECUTION_ORDER
        ]
    }


def manifest_filename(
    model_id: str,
    config_name: str,
    dataset_path: str = DATASET_PATH,
    split: str = SPLIT,
) -> str:
    if config_name not in {item.config_name for item in SCORE_CONFIGS}:
        raise ValueError(f"Unknown multilingual benchmark config: {config_name}")
    dataset_slug = dataset_path.replace("/", "-")
    return (
        f"MODEL_{model_slug(model_id)}_DATASET_{dataset_slug}_"
        f"{config_name}_{split}.jsonl"
    )


def score_manifest(manifest_path: Path, language: str) -> float:
    from jiwer import wer

    from normalizer import data_utils
    from normalizer.eval_utils import normalize_compound_pairs, read_manifest

    manifest = read_manifest(str(manifest_path))
    if not manifest:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    references = [
        data_utils.ml_normalizer(item["text"], lang=language) for item in manifest
    ]
    predictions = [
        data_utils.ml_normalizer(item["pred_text"], lang=language) for item in manifest
    ]
    references, predictions = normalize_compound_pairs(references, predictions)
    return round(100 * wer(references, predictions), 2)


def score_results(
    results_dir: Path,
    model_id: str,
    dataset_path: str = DATASET_PATH,
    split: str = SPLIT,
    rtfx: str = "-1",
) -> tuple[list[str], list[str], list[tuple[str, float]]]:
    validate_model_id(model_id)
    scores: list[tuple[str, float]] = []
    for config in SCORE_CONFIGS:
        manifest_path = results_dir / manifest_filename(
            model_id, config.config_name, dataset_path, split
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing result manifest: {manifest_path}")
        scores.append((config.column, score_manifest(manifest_path, config.language)))

    average = sum(score for _, score in scores) / len(scores)
    header = [
        "model",
        "Model size (B)",
        "RTFx",
        *(config.column for config in SCORE_CONFIGS),
        "Avg",
    ]
    row = [model_id, "", rtfx, *(str(score) for _, score in scores), str(average)]
    return header, row, scores


def write_csv_rows(path: Path, rows: Iterable[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        csv.writer(output, lineterminator="\n").writerows(rows)


def collect_result_artifacts(
    artifact_root: Path, results_dir: Path, model_id: str
) -> list[Path]:
    """Select the newest immutable run-attempt artifact for every config."""
    slug = model_slug(model_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    children = list(artifact_root.iterdir()) if artifact_root.is_dir() else []
    for config in SCORE_CONFIGS:
        pattern = re.compile(
            rf"^multilingual-results-(\d+)-{re.escape(slug)}-"
            rf"{re.escape(config.config_name)}$"
        )
        candidates = []
        for child in children:
            match = pattern.fullmatch(child.name)
            if match and child.is_dir():
                manifest = child / manifest_filename(model_id, config.config_name)
                if manifest.is_file():
                    candidates.append((int(match.group(1)), manifest))
        if not candidates:
            continue
        attempt, source = max(candidates, key=lambda candidate: candidate[0])
        destination = results_dir / source.name
        shutil.copy2(source, destination)
        print(f"Selected run attempt {attempt} for {config.config_name}: {source.name}")
        collected.append(destination)
    return collected


def prepare_command(args: argparse.Namespace) -> None:
    validate_model_id(args.model_id)
    validate_env_name(args.api_key_env, "api_key_env")
    validate_env_name(args.api_key_secret, "api_key_secret")
    validate_range(args.max_workers, "max_workers", 1, 512)
    validate_range(args.max_parallel, "max_parallel", 1, len(SCORE_CONFIGS))
    validate_range(args.max_samples, "max_samples", 0, 1_000_000)
    validate_run_id(args.reuse_run_id, "reuse_run_id")
    validate_run_id(args.result_run_id, "result_run_id")

    output_lines = (
        f"model_slug={model_slug(args.model_id)}\n"
        f"matrix={json.dumps(matrix_payload(), separators=(',', ':'))}\n"
    )
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(output_lines)
    else:
        print(output_lines, end="")


def score_command(args: argparse.Namespace) -> None:
    header, row, scores = score_results(
        results_dir=args.results_dir,
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        split=args.split,
        rtfx=args.rtfx,
    )
    for column, score in scores:
        print(f"{column}: {score:.2f}")
    write_csv_rows(args.output, (header, row))
    print(",".join(header))
    print(",".join(row))


def collect_command(args: argparse.Namespace) -> None:
    collected = collect_result_artifacts(
        args.artifact_root, args.results_dir, args.model_id
    )
    print(f"Collected {len(collected)} result manifests")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Validate inputs and emit matrix outputs"
    )
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--api-key-env", required=True)
    prepare.add_argument("--api-key-secret", required=True)
    prepare.add_argument("--max-workers", type=int, required=True)
    prepare.add_argument("--max-parallel", type=int, required=True)
    prepare.add_argument("--max-samples", type=int, required=True)
    prepare.add_argument("--reuse-run-id", default="")
    prepare.add_argument("--result-run-id", default="")
    prepare.add_argument("--github-output", type=Path)
    prepare.set_defaults(handler=prepare_command)

    score = subparsers.add_parser("score", help="Score all multilingual manifests")
    score.add_argument("--results-dir", type=Path, required=True)
    score.add_argument("--model-id", required=True)
    score.add_argument("--dataset-path", default=DATASET_PATH)
    score.add_argument("--split", default=SPLIT)
    score.add_argument("--rtfx", default="-1")
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=score_command)

    collect = subparsers.add_parser(
        "collect", help="Collect the newest result artifact for each config"
    )
    collect.add_argument("--artifact-root", type=Path, required=True)
    collect.add_argument("--results-dir", type=Path, required=True)
    collect.add_argument("--model-id", required=True)
    collect.set_defaults(handler=collect_command)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    try:
        parsed_args.handler(parsed_args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
