import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jiwer import wer

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs, read_manifest


# The leaderboard labels the Mozilla Common Voice source splits as CoVoST.
CONFIG_COLUMNS = (
    ("mcv_de", "de_covost", "de"),
    ("fleurs_de", "de_fleurs", "de"),
    ("mcv_fr", "fr_covost", "fr"),
    ("mls_fr", "fr_mls", "fr"),
    ("fleurs_fr", "fr_fleurs", "fr"),
    ("mcv_it", "it_covost", "it"),
    ("mls_it", "it_mls", "it"),
    ("fleurs_it", "it_fleurs", "it"),
    ("mcv_es", "es_covost", "es"),
    ("mls_es", "es_mls", "es"),
    ("fleurs_es", "es_fleurs", "es"),
    ("mls_pt", "pt_mls", "pt"),
    ("fleurs_pt", "pt_fleurs", "pt"),
)


def score_manifest(manifest_path: Path, language: str) -> float:
    manifest = read_manifest(str(manifest_path))
    references = [
        data_utils.ml_normalizer(item["text"], lang=language) for item in manifest
    ]
    predictions = [
        data_utils.ml_normalizer(item["pred_text"], lang=language)
        for item in manifest
    ]
    references, predictions = normalize_compound_pairs(references, predictions)
    return round(100 * wer(references, predictions), 2)


def score_multilingual_results(
    results_dir: Path,
    model_id: str,
    dataset_path: str,
    split: str,
    rtfx: str,
) -> str:
    model_slug = model_id.replace("/", "-")
    dataset_slug = dataset_path.replace("/", "-")
    scores = []

    for config_name, column, language in CONFIG_COLUMNS:
        manifest_path = results_dir / (
            f"MODEL_{model_slug}_DATASET_{dataset_slug}_{config_name}_{split}.jsonl"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing result manifest: {manifest_path}")
        score = score_manifest(manifest_path, language)
        scores.append(score)
        print(f"{column}: {score:.2f}")

    average = sum(scores) / len(scores)
    header = (
        "model,Model size (B),RTFx,"
        + ",".join(column for _, column, _ in CONFIG_COLUMNS)
        + ",Avg"
    )
    row = ",".join(
        [model_id, "", rtfx]
        + [str(score) for score in scores]
        + [str(average)]
    )
    print(header)
    print(row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score all multilingual manifests and print a leaderboard CSV row."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--dataset-path", default="nithinraok/asr-leaderboard-datasets"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--rtfx",
        default="-1",
        help="RTFx CSV value. API submissions conventionally use -1.",
    )
    args = parser.parse_args()
    score_multilingual_results(
        results_dir=args.results_dir,
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        split=args.split,
        rtfx=args.rtfx,
    )
