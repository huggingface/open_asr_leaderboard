import argparse
import csv
import os
from pathlib import Path
import tempfile


def read_score_row(scores_path: Path, model_id: str) -> list[str]:
    prefix = f"{model_id},"
    lines = [
        line.strip()
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [line for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one score row for {model_id}, found {len(matches)}"
        )
    return next(csv.reader([matches[0]]))


def update_csv(csv_path: Path, scores_path: Path, model_id: str) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    if not rows or rows[0][0] != "model":
        raise ValueError(f"Unexpected multilingual CSV header in {csv_path}")

    score_row = read_score_row(scores_path, model_id)
    if len(score_row) != len(rows[0]):
        raise ValueError(
            f"Score row has {len(score_row)} columns; expected {len(rows[0])}"
        )

    matching_indices = [
        index for index, row in enumerate(rows[1:], start=1) if row and row[0] == model_id
    ]
    if len(matching_indices) > 1:
        raise ValueError(f"Multiple existing rows found for {model_id}")
    if matching_indices:
        rows[matching_indices[0]] = score_row
    else:
        rows.append(score_row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows(rows)
        os.replace(temporary_name, csv_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Insert or replace a scored model row in multilingual.csv."
    )
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--scores-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    arguments = parser.parse_args()
    update_csv(arguments.csv_path, arguments.scores_path, arguments.model_id)
    print(f"Updated {arguments.csv_path} for {arguments.model_id}")
