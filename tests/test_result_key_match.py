"""Model ids that share a prefix must not share a CSV cell."""

import importlib.util
import sys
import types
from pathlib import Path

_kaldialign = types.ModuleType("kaldialign")
_kaldialign.edit_distance = lambda *args, **kwargs: None
sys.modules.setdefault("kaldialign", _kaldialign)

_path = Path(__file__).resolve().parents[1] / "normalizer" / "eval_utils.py"
_spec = importlib.util.spec_from_file_location("eval_utils_under_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
result_key_matches_model = _module.result_key_matches_model


def test_prefix_model_does_not_match_longer_id():
    turbo = "openai/whisper-large-v3-turbo | librispeech_asr_clean_test"
    base = "openai/whisper-large-v3 | librispeech_asr_clean_test"
    assert result_key_matches_model("openai/whisper-large-v3", base)
    assert not result_key_matches_model("openai/whisper-large-v3", turbo)
    assert result_key_matches_model("openai/whisper-large-v3-turbo", turbo)
    assert result_key_matches_model(" openai/whisper-large-v3 ", base)
