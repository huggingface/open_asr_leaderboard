"""Exercise both runners' real dispatch statements without GPU/model downloads.

Run with: python -m unittest discover -s tests -p 'test_qwen3_asr_detection.py' -v
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TokenBatch:
    def __init__(self):
        self.slice = None

    def __getitem__(self, key):
        self.slice = key
        return self


class RecordingProcessor:
    def __init__(self):
        self.kwargs = None

    def decode(self, tokens, **kwargs):
        self.kwargs = kwargs
        # Model the structured-output contract, including a multilingual tag.
        if kwargs.get("return_format") == "transcription_only":
            return ["bonjour le monde"]
        return ["language French<asr_text>bonjour le monde"]

    batch_decode = decode


def run_dispatch(filename, model_id, config):
    tree = ast.parse((ROOT / "transformers" / filename).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    detection = next(n for n in main.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "is_qwen3_asr" for t in n.targets))
    benchmark = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark")
    # Select the complete decoding if/elif chain, not a copied predicate.
    decode = next(n for n in benchmark.body if isinstance(n, ast.If)
                  and any(isinstance(x, ast.Assign) and any(
                      isinstance(t, ast.Name) and t.id == "pred_text" for t in x.targets
                  ) for x in ast.walk(n)))
    processor, tokens = RecordingProcessor(), TokenBatch()
    env = {"args": SimpleNamespace(model_id=model_id), "config": config,
           "processor": processor, "pred_ids": tokens, "prompt_len": 7,
           "has_transcription_processor": True, "texts": None}
    for node in ast.walk(decode):
        if isinstance(node, ast.Name) and node.id.startswith("is_"):
            env[node.id] = False
    exec(compile(ast.Module(body=[detection, decode], type_ignores=[]), filename, "exec"), env)
    return env["is_qwen3_asr"], env["pred_text"], processor.kwargs, tokens.slice


class Qwen3AsrDispatchTest(unittest.TestCase):
    def test_config_only_finetune_uses_structured_decode_and_strips_prompt(self):
        for runner in ("run_eval.py", "run_eval_ml.py"):
            with self.subTest(runner=runner):
                detected, text, kwargs, sliced = run_dispatch(
                    runner, "oruk/orukqwen-asr-1.7b", SimpleNamespace(model_type="qwen3_asr"))
                self.assertTrue(detected)
                self.assertEqual(text, ["bonjour le monde"])
                self.assertEqual(kwargs, {"return_format": "transcription_only"})
                self.assertEqual(sliced, (slice(None), slice(7, None)))

    def test_legacy_name_fallback_remains_case_insensitive(self):
        for runner in ("run_eval.py", "run_eval_ml.py"):
            with self.subTest(runner=runner):
                detected, text, _, _ = run_dispatch(runner, "Qwen/QWEN3-ASR-1.7B-hf", SimpleNamespace())
                self.assertTrue(detected)
                self.assertEqual(text, ["bonjour le monde"])

    def test_unrelated_models_keep_generic_decoding(self):
        for runner in ("run_eval.py", "run_eval_ml.py"):
            for config in (SimpleNamespace(), SimpleNamespace(model_type="other_asr")):
                with self.subTest(runner=runner, config=config):
                    detected, text, kwargs, _ = run_dispatch(runner, "example/audio-model", config)
                    self.assertFalse(detected)
                    self.assertEqual(kwargs, {"skip_special_tokens": True})
                    self.assertEqual(text, ["language French<asr_text>bonjour le monde"])


if __name__ == "__main__":
    unittest.main()
