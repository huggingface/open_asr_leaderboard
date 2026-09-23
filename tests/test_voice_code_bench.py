import copy
import json
from argparse import Namespace

import pytest
from datasets import Audio, Dataset

from normalizer import data_utils, voice_code_bench as vcb
from normalizer.eval_utils import read_manifest, score_results, write_manifest
from voice_code_bench.verifier import (
    VerifierCache,
    build_cache_entry,
    load_verifier_config,
    verifier_cache_key,
    verifier_input,
)


@pytest.fixture
def scoring_case(tmp_path):
    metadata = []
    manifest = []
    cache = VerifierCache(tmp_path / "verifier.json")
    config = load_verifier_config(vcb.VERIFIER_ID)
    # Different entity counts distinguish pooled CTEM (25%) from a per-row
    # average (50%). An empty prediction must count as missed entities.
    for audio_id, count, transcript in [("one", 1, "--dry-run"), ("two", 3, "")]:
        entities = [{
            "id": f"{audio_id}_{i}", "type": "cli_flag", "role": "option",
            "acoustic": "double dash dry dash run", "canonical": "--dry-run",
        } for i in range(count)]
        row = {"audio_id": audio_id, "entities": entities}
        metadata.append(row)
        manifest.append({
            "audio_id": audio_id, "pred_text": transcript,
            "voice_code_bench_revision": vcb.DATASET_REVISION,
        })
        request = verifier_input(row, transcript)
        response = {"entities": [{
            "target_index": i, "type": entity["type"], "canonical": entity["canonical"],
            "present": bool(transcript), "evidence": transcript,
            "reason": "Exact flag present." if transcript else "Empty transcript.",
        } for i, entity in enumerate(entities)]}
        cache.put(build_cache_entry(config, request, verifier_cache_key(config, request), response))
    return metadata, manifest, cache


def test_pooled_ctem_and_sorting_use_ids(scoring_case):
    metadata, manifest, cache = scoring_case
    report = vcb.score_manifest(list(reversed(manifest)), metadata, cache=cache)
    assert report["ctem_percent"] == 25
    assert report["correct_entities"] == 1
    assert report["total_entities"] == 4
    assert report["complete_test_split"] is True
    assert report["higher_is_better"] is True
    assert report["entity_scores"][0]["datapoint_id"] == "two"
    assert "wer" not in report


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "unknown", "revision", "text", "empty"])
def test_invalid_manifests_fail_before_verification(scoring_case, monkeypatch, invalid):
    metadata, manifest, cache = scoring_case
    manifest = copy.deepcopy(manifest)
    if invalid == "duplicate":
        manifest.append(manifest[0])
    elif invalid == "missing":
        manifest.pop()
    elif invalid == "unknown":
        manifest[0]["audio_id"] = "unknown"
    elif invalid == "revision":
        manifest[0]["voice_code_bench_revision"] = "main"
    elif invalid == "text":
        manifest[0]["pred_text"] = None
    elif invalid == "empty":
        manifest = []

    def fail(*args, **kwargs):
        pytest.fail("Verifier called before input validation completed")

    monkeypatch.setattr("voice_code_bench.verifier.verify_entity_matches", fail)
    with pytest.raises(ValueError):
        vcb.score_manifest(manifest, metadata, cache=cache)


def test_partial_run_requires_explicit_opt_in(scoring_case):
    metadata, manifest, cache = scoring_case
    report = vcb.score_manifest(manifest[:1], metadata, cache=cache, allow_partial=True)
    assert report["complete_test_split"] is False
    assert report["recordings"] == 1
    assert report["ctem_percent"] == 100


def test_replay_does_not_call_provider_and_does_not_normalize(scoring_case, monkeypatch):
    metadata, manifest, cache = scoring_case

    def fail(*args, **kwargs):
        pytest.fail("Replay must not call the provider")

    monkeypatch.setattr("voice_code_bench.verifier.call_verifier_provider", fail)
    vcb.score_manifest(manifest, metadata, cache=cache)
    manifest[0]["pred_text"] = "dry run"
    with pytest.raises(RuntimeError, match="cache miss"):
        vcb.score_manifest(manifest, metadata, cache=cache)


def test_loader_attaches_pinned_audio_and_preserves_annotations(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    (root / "data/audio").mkdir(parents=True)
    (root / "data/audio/001.wav").touch()
    source = {
        "file_name": "audio/001.wav", "audio_id": "work_001",
        "transcripts": {"acoustic": "Use --dry-run and DATABASE_URL."},
        "entities": [{"canonical": "--dry-run"}],
    }
    (root / "data/metadata.jsonl").write_text(json.dumps(source) + "\n")

    def snapshot(repo_id, **kwargs):
        assert repo_id == vcb.DATASET_ID
        assert kwargs["revision"] == vcb.DATASET_REVISION
        assert kwargs["allow_patterns"] == ["data/metadata.jsonl", "data/audio/*.wav"]
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    args = Namespace(dataset_path=vcb.DATASET_ID, dataset="default", split="test", streaming=False)
    dataset = data_utils.load_data(args).cast_column("audio", Audio(decode=False))
    row = dataset[0]
    assert row["audio"]["path"] == str(root / "data/audio/001.wav")
    assert row["id"] == row["audio_id"] == "work_001"
    assert row["text"] == source["transcripts"]["acoustic"]
    assert row["entities"] == source["entities"]
    assert row["voice_code_bench_revision"] == vcb.DATASET_REVISION


@pytest.mark.parametrize("field,value", [("split", "train"), ("dataset", "other"), ("streaming", True)])
def test_loader_rejects_unsupported_modes(field, value):
    args = Namespace(dataset="default", split="test", streaming=False)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        vcb.load_data(args)


def test_prepare_data_bypasses_normalization_and_filtering(monkeypatch):
    dataset = Dataset.from_dict({"text": ["--dry-run DATABASE_URL", ""], "audio": [None, None]})

    def fail(*args, **kwargs):
        pytest.fail("VCB must not apply the WER normalizer")

    monkeypatch.setattr(data_utils, "normalizer", fail)
    result = data_utils.prepare_data(dataset, normalize_text=False)
    assert list(result["original_text"]) == ["--dry-run DATABASE_URL", ""]


def test_wer_path_still_normalizes_and_filters():
    dataset = Dataset.from_dict({"text": ["HELLO!", ""], "audio": [None, None]})
    result = data_utils.prepare_data(dataset)
    assert list(result["norm_text"]) == ["hello"]


def test_manifest_preserves_sorted_ids_and_generic_scorer_refuses_ctem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_manifest(
        ["ref two", "ref one"], ["", "--dry-run"], "nvidia/parakeet-tdt-0.6b-v3",
        vcb.DATASET_ID, "default", "test",
        extra_fields={"audio_id": ["two", "one"], "voice_code_bench_revision": [vcb.DATASET_REVISION] * 2},
    )
    rows = read_manifest(path)
    assert [row["audio_id"] for row in rows] == ["two", "one"]
    assert rows[1]["pred_text"] == "--dry-run"
    with pytest.raises(ValueError, match="VoiceCodeBench uses CTEM"):
        score_results("results")
