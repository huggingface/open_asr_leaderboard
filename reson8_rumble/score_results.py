"""Validate and score the downloaded manifests from one evaluation run."""

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

from reson8_rumble.dataset_config import manifest_name, parse_datasets


def score_run(results_dir, model_id, datasets):
    from normalizer.eval_utils import score_results

    selections = parse_datasets(datasets)
    expected = {
        manifest_name(model_id, path, config, split): language
        for path, config, split, language in selections
    }
    manifests = list(results_dir.rglob("*.jsonl"))
    if len(manifests) != len(selections):
        raise ValueError(
            f"Expected {len(selections)} manifests, found {len(manifests)}; "
            "refusing to score a partial run"
        )
    found = {path.name for path in manifests}
    if found != expected.keys():
        raise ValueError(
            f"Manifests do not match the selected datasets: "
            f"missing={sorted(expected.keys() - found)}, "
            f"unexpected={sorted(found - expected.keys())}"
        )
    by_language = defaultdict(list)
    for path in manifests:
        by_language[expected[path.name]].append(path)

    # The shared scorer applies one language to every file in its directory.
    # Give each pass only that language's manifests, without filtering families.
    for language, paths in sorted(by_language.items()):
        with tempfile.TemporaryDirectory(prefix="rumble-score-") as directory:
            for path in paths:
                (Path(directory) / path.name).symlink_to(path.resolve())
            score_results(
                directory, model_id, language=language, multilingual=language != "en",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("model_id")
    parser.add_argument("--datasets", required=True)
    args = parser.parse_args()
    try:
        score_run(args.results_dir, args.model_id, args.datasets)
    except ValueError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
