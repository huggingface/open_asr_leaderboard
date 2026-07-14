import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from api import run_eval_ml


class FakeDataset(list):
    def cast_column(self, *_args, **_kwargs):
        return self

    def select(self, indices):
        return FakeDataset(self[index] for index in indices)


def sample(text):
    return {"text": text, "audio": {"array": [0.0] * 16, "sampling_rate": 16}}


class RunEvalResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_directory = os.getcwd()
        os.chdir(self.temporary_directory.name)
        self.dataset = FakeDataset([sample("first"), sample("second")])
        self.metric = mock.Mock()
        self.metric.compute.return_value = 0.0
        self.patches = [
            mock.patch.object(
                run_eval_ml.datasets, "load_dataset", return_value=self.dataset
            ),
            mock.patch.object(run_eval_ml, "Audio", return_value=object()),
            mock.patch.object(run_eval_ml.sf, "write"),
            mock.patch.object(run_eval_ml.evaluate, "load", return_value=self.metric),
            mock.patch.object(
                run_eval_ml.data_utils, "is_target_text_in_range", return_value=True
            ),
            mock.patch.object(
                run_eval_ml.data_utils,
                "ml_normalizer",
                side_effect=lambda value, **_: value,
            ),
            mock.patch.object(
                run_eval_ml,
                "normalize_compound_pairs",
                side_effect=lambda refs, preds: (refs, preds),
            ),
            mock.patch.object(
                run_eval_ml.data_utils,
                "write_manifest",
                return_value=Path("result.jsonl"),
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        os.chdir(self.previous_directory)
        self.temporary_directory.cleanup()

    def run_transcription(self, **overrides):
        options = dict(
            dataset_path="owner/dataset",
            config_name="fleurs_de",
            split="test",
            model_name="fake/model",
            language="de",
            max_workers=2,
            resume=True,
        )
        options.update(overrides)
        run_eval_ml.transcribe_dataset(**options)

    def test_concurrent_results_are_written_in_dataset_order(self):
        def transcribe(_model, _path, current_sample, **_kwargs):
            if current_sample["text"] == "first":
                time.sleep(0.02)
            return current_sample["text"].upper()

        with mock.patch.object(
            run_eval_ml, "transcribe_with_retry", side_effect=transcribe
        ):
            self.run_transcription()

        write_manifest = run_eval_ml.data_utils.write_manifest
        self.assertEqual(write_manifest.call_args.args[0], ["first", "second"])
        self.assertEqual(write_manifest.call_args.args[1], ["FIRST", "SECOND"])
        self.assertFalse(any(Path("results/.checkpoints").glob("*.partial")))

    def test_seed_manifest_avoids_duplicate_provider_call(self):
        seed_path = Path("seed.jsonl")
        seed_path.write_text(
            json.dumps(
                {"text": "first", "pred_text": "FIRST", "duration": 1, "time": 0.1}
            )
            + "\n",
            encoding="utf-8",
        )
        with mock.patch.object(
            run_eval_ml, "transcribe_with_retry", return_value="SECOND"
        ) as transcribe:
            self.run_transcription(seed_manifest=str(seed_path))

        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(transcribe.call_args.args[2]["text"], "second")

    def test_failure_keeps_successful_samples_in_checkpoint(self):
        def transcribe(_model, _path, current_sample, **_kwargs):
            if current_sample["text"] == "second":
                raise RuntimeError("provider failed")
            return "FIRST"

        with (
            mock.patch.object(
                run_eval_ml, "transcribe_with_retry", side_effect=transcribe
            ),
            self.assertRaisesRegex(RuntimeError, "1 samples failed"),
        ):
            self.run_transcription()

        checkpoints = list(Path("results/.checkpoints").glob("*.partial"))
        self.assertEqual(len(checkpoints), 1)
        records = [
            json.loads(line)
            for line in checkpoints[0].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["index"] for record in records], [0])


if __name__ == "__main__":
    unittest.main()
