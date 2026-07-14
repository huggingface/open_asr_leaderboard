import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from api import multilingual_benchmark as benchmark


class MultilingualBenchmarkTests(unittest.TestCase):
    def test_matrix_and_score_spec_cover_the_same_configs(self):
        matrix = benchmark.matrix_payload()["include"]
        self.assertEqual(
            [item["config"] for item in matrix], list(benchmark.EXECUTION_ORDER)
        )
        self.assertEqual(
            {item["config"] for item in matrix},
            {item.config_name for item in benchmark.SCORE_CONFIGS},
        )
        self.assertEqual(len(matrix), 13)

    def test_model_and_secret_inputs_are_path_safe(self):
        self.assertEqual(
            benchmark.model_slug("assembly/universal-3-pro"),
            "assembly-universal-3-pro",
        )
        for invalid in ("assembly", "../model", "assembly/model/name", "a/ model"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark.validate_model_id(invalid)
        for invalid in ("secret", "A-B", "1SECRET", "A=B"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark.validate_env_name(invalid, "secret")

    def test_prepare_emits_compact_github_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "github-output.txt"
            args = argparse.Namespace(
                model_id="assembly/universal-3-pro",
                api_key_env="ASSEMBLYAI_API_KEY",
                api_key_secret="ASSEMBLYAI_API_KEY",
                max_workers=8,
                max_parallel=1,
                max_samples=1,
                reuse_run_id="123",
                result_run_id="",
                github_output=output,
            )
            benchmark.prepare_command(args)
            values = dict(
                line.split("=", 1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(values["model_slug"], "assembly-universal-3-pro")
            self.assertEqual(len(json.loads(values["matrix"])["include"]), 13)

    def test_score_results_uses_leaderboard_order_and_rounded_mean(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            for config in benchmark.SCORE_CONFIGS:
                (
                    results_dir
                    / benchmark.manifest_filename(
                        "openai/whisper-1", config.config_name
                    )
                ).touch()

            values = [float(index) for index in range(1, 14)]
            with mock.patch.object(
                benchmark, "score_manifest", side_effect=values
            ) as score_manifest:
                header, row, scores = benchmark.score_results(
                    results_dir, "openai/whisper-1"
                )

            self.assertEqual(score_manifest.call_count, 13)
            self.assertEqual(
                header[3:-1], [config.column for config in benchmark.SCORE_CONFIGS]
            )
            self.assertEqual(row[0:3], ["openai/whisper-1", "", "-1"])
            self.assertEqual(float(row[-1]), sum(values) / len(values))
            self.assertEqual([column for column, _ in scores], header[3:-1])

    def test_score_results_fails_closed_when_a_manifest_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing result manifest"):
                benchmark.score_results(Path(temporary_directory), "openai/whisper-1")

    def test_score_manifest_uses_repository_normalizer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "result.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "text": "Hallo, Welt!",
                        "pred_text": "hallo welt",
                        "duration": 1.0,
                        "time": 0.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(benchmark.score_manifest(manifest, "de"), 0.0)

    def test_csv_writer_preserves_empty_model_size_column(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scores.csv"
            benchmark.write_csv_rows(path, (["model", "size"], ["a/b", ""]))
            with path.open(encoding="utf-8", newline="") as source:
                self.assertEqual(
                    list(csv.reader(source)), [["model", "size"], ["a/b", ""]]
                )

    def test_collect_result_artifacts_prefers_newest_run_attempt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = root / "artifacts"
            results = root / "results"
            config = benchmark.SCORE_CONFIGS[0]
            filename = benchmark.manifest_filename(
                "assembly/universal-3-pro", config.config_name
            )
            for attempt, contents in ((1, "old"), (3, "new"), (2, "middle")):
                directory = artifacts / (
                    "multilingual-results-"
                    f"{attempt}-assembly-universal-3-pro-{config.config_name}"
                )
                directory.mkdir(parents=True)
                (directory / filename).write_text(contents, encoding="utf-8")

            collected = benchmark.collect_result_artifacts(
                artifacts, results, "assembly/universal-3-pro"
            )

            self.assertEqual(collected, [results / filename])
            self.assertEqual((results / filename).read_text(encoding="utf-8"), "new")


if __name__ == "__main__":
    unittest.main()
