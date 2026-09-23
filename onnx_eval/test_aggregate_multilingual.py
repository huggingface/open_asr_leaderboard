import json
import tempfile
import unittest
from pathlib import Path

from aggregate_multilingual import CONFIG_TO_COLUMN, build_rows, load_metrics, parse_variant_specs


class AggregateMultilingualTest(unittest.TestCase):
    def test_builds_weighted_rtfx_and_expected_columns(self):
        specs = parse_variant_specs(["fp32=owner/fp32,0.6", "int8=owner/int8,0.6"])
        metrics = []
        for variant, model_id in (("fp32", "owner/fp32"), ("int8", "owner/int8")):
            for index, config in enumerate(CONFIG_TO_COLUMN):
                metrics.append(
                    {
                        "variant": variant,
                        "model_id": model_id,
                        "config_name": config,
                        "processed_samples": 10,
                        "total_audio_seconds": 100.0,
                        "total_inference_seconds": 10.0 if variant == "fp32" else 5.0,
                        "wer": float(index + 1),
                    }
                )

        rows, summary = build_rows(metrics, specs)

        self.assertEqual(rows[0]["model"], "owner/fp32")
        self.assertEqual(rows[0]["RTFx"], "10")
        self.assertEqual(rows[1]["RTFx"], "20")
        self.assertEqual(rows[0]["de_covost"], "1")
        self.assertEqual(rows[0]["Avg"], "7")
        self.assertEqual(summary["variants"]["int8"]["processed_samples"], 130)

    def test_load_metrics_is_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "artifact" / "result"
            nested.mkdir(parents=True)
            (nested / "metrics.json").write_text(json.dumps({"variant": "fp32"}), encoding="utf-8")
            self.assertEqual(load_metrics(Path(directory)), [{"variant": "fp32"}])


if __name__ == "__main__":
    unittest.main()
