"""Contracts for the open-weight Qwen cutover; all network calls are mocked."""

from contextlib import nullcontext
from copy import deepcopy
import json
import threading
import time
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from normalizer import voice_code_bench as scorer
from normalizer import voice_code_bench_judge as judge


@pytest.fixture
def point():
    return {"audio_id": "fixture", "entities": [
        {"id": "port", "type": "port", "role": "port", "canonical": "7326",
         "acoustic": "seven three two six", "alternative_spellings": []}]}


def response(request, present=True):
    data = json.loads(request["messages"][1]["content"])
    entries = {}
    for target in data["target_entities"]:
        entries[str(target["target_index"])] = {
            **{key: target[key] for key in ("target_index", "type", "canonical")},
            "evidence_start_segment": 0 if present else -1,
            "evidence_end_segment": 0 if present else -1,
            "reason": "Complete retained value." if present else "Value absent.", "present": present}
    return {"model": judge.MODEL, "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"entities": entries})}}]}


def test_request_retains_raw_unicode_and_tested_configuration(point):
    text = "a" * 255 + "é/--" + "a" * 260
    request = judge.build_request(point, text)
    data = json.loads(request["messages"][1]["content"])
    assert data["transcript"] == text
    assert "".join(segment["text"] for segment in data["transcript_segments"]) == text
    assert data["target_entities"][0]["acoustic"] == point["entities"][0]["acoustic"]
    assert request["model"] == "Qwen/Qwen3.8-27B"
    assert request["thinking_token_budget"] == 8192
    assert request["max_tokens"] == 16384
    assert request["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    assert "enable_thinking" not in request
    assert "thinking_budget" not in request


def test_pinned_metadata_explicitly_has_no_alternative_spellings(point):
    original = deepcopy(point)
    del original["entities"][0]["alternative_spellings"]
    with patch("voice_code_bench.io.read_metadata", return_value=[original]):
        assert scorer.load_metadata("fixture.jsonl") == [point]
    assert "alternative_spellings" not in original["entities"][0]


@pytest.mark.parametrize("mutation", [
    lambda entry: entry.update(target_index=True),
    lambda entry: entry.update(target_index=0.0),
    lambda entry: entry.update(canonical="7327"),
    lambda entry: entry.update(type="phone_number"),
    lambda entry: entry.update(present="true"),
    lambda entry: entry.update(evidence_start_segment=True),
    lambda entry: entry.update(evidence_start_segment=0.0),
    lambda entry: entry.update(evidence_end_segment=1),
    lambda entry: entry.update(evidence_start_segment=-1),
    lambda entry: entry.update(reason=" "),
    lambda entry: entry.update(extra="unexpected"),
])
def test_rejects_corrupt_identity_verdicts_and_ranges(point, mutation):
    request = judge.build_request(point, "port 7326")
    raw = response(request)
    parsed = json.loads(raw["choices"][0]["message"]["content"])
    mutation(parsed["entities"]["0"])
    raw["choices"][0]["message"]["content"] = json.dumps(parsed)
    with pytest.raises(ValueError, match="Invalid Qwen response"):
        judge.parse_response(raw, request)


def test_rejects_missing_duplicate_truncated_and_empty_positive_outputs(point):
    request = judge.build_request(point, "port 7326")
    for content in ('{"entities":{}}', '{"entities":{},"entities":{}}', '[]', 'invalid'):
        raw = response(request)
        raw["choices"][0]["message"]["content"] = content
        with pytest.raises(ValueError):
            judge.parse_response(raw, request)
    raw = response(request)
    raw["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        judge.parse_response(raw, request)
    empty = judge.build_request(point, "")
    with pytest.raises(ValueError):
        judge.parse_response(response(empty), empty)


@pytest.mark.parametrize("present", [True, False])
def test_single_call_and_credential_free_replay(tmp_path, point, present):
    cache = judge.VerifierCache(tmp_path / "qwen27b.json")
    original = deepcopy(point)
    with patch.object(judge, "post_request", side_effect=lambda request: response(request, present)) as post:
        matches = judge.verify_entity_matches(point, "port 7326", cache=cache, cache_mode="live-fill")
    post.assert_called_once()
    assert matches[0]["present"] is present
    assert point == original
    saved = judge.VerifierCache(cache.path)
    assert len(saved.responses) == 1
    assert all(raw["usage"]["prompt_tokens"] == 12 for raw in saved.responses.values())
    manifest = [{"audio_id": point["audio_id"], "pred_text": "port 7326"}]
    with patch.dict("os.environ", {}, clear=True), patch.object(judge, "post_request") as post, \
         patch.object(judge, "local_runtime") as runtime:
        assert judge.verify_manifest(manifest, {point["audio_id"]: point},
                                     cache=saved, cache_mode="replay", workers=16) == [matches]
        runtime.assert_not_called()
        post.assert_not_called()


def test_failed_call_is_not_cached_or_retried_and_replay_does_not_guess(tmp_path, point):
    cache = judge.VerifierCache(tmp_path / "qwen27b.json")
    with patch.object(judge, "post_request", side_effect=RuntimeError("Provider failure")) as call:
        with pytest.raises(RuntimeError):
            judge.verify_entity_matches(point, "port 7326", cache=cache, cache_mode="live-fill")
    call.assert_called_once()
    assert not cache.responses
    with patch.object(judge, "post_request") as call:
        with pytest.raises(ValueError, match="replay cache miss"):
            judge.verify_entity_matches(point, "port 7326", cache=cache, cache_mode="replay")
        call.assert_not_called()


def test_legacy_wrong_config_and_poisoned_caches_fail(tmp_path, point):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": {}}))
    with pytest.raises(ValueError, match="Incompatible verifier cache"):
        judge.VerifierCache(path)
    path.unlink()
    cache = judge.VerifierCache(path)
    request = judge.build_request(point, "port 7326")
    digest = judge.request_digest(request)
    cache.put(digest, {"choices": []})
    with patch.object(judge, "post_request") as post:
        with pytest.raises(ValueError):
            judge.verify_entity_matches(point, "port 7326", cache=cache, cache_mode="live-fill")
        post.assert_not_called()
    data = json.loads(path.read_text())
    data["config_digest"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incompatible verifier cache"):
        judge.VerifierCache(path)


def test_http_failure_is_one_call_and_does_not_expose_body():
    error = urllib.error.HTTPError(judge.ENDPOINT, 429, "private response", {}, None)
    with patch("urllib.request.urlopen", side_effect=error) as call:
        with pytest.raises(RuntimeError, match="HTTP 429; no automatic retry"):
            judge.post_request({})
    call.assert_called_once()


def test_workers_bound_requests_and_preserve_manifest_order(tmp_path, point):
    lock = threading.Lock()
    active = maximum = 0
    manifest = [{"audio_id": str(i), "pred_text": str(i)} for i in range(8)]
    targets = {str(i): {**point, "audio_id": str(i)} for i in range(8)}
    def post(request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return response(request)
    cache = judge.VerifierCache(tmp_path / "cache.json")
    with patch.object(judge, "local_runtime", return_value=nullcontext()), patch.object(judge, "post_request", side_effect=post):
        matches = judge.verify_manifest(manifest, targets, cache=cache, cache_mode="live-fill", workers=2)
    assert maximum == 2
    assert len(judge.VerifierCache(cache.path).responses) == 8
    assert [match[0]["evidence"] for match in matches] == [str(i) for i in range(8)]


def test_entire_manifest_validated_before_gpu_start(tmp_path, point):
    manifest = [{"audio_id": "fixture", "pred_text": "port 7326", "voice_code_bench_revision": scorer.DATASET_REVISION}]
    cache = judge.VerifierCache(tmp_path / "cache.json")
    with patch.object(judge, "post_request") as post:
        with pytest.raises(ValueError, match="duplicate"):
            scorer.score_manifest(manifest * 2, [point], cache=cache, cache_mode="live-fill")
        post.assert_not_called()


def test_ctem_keeps_empty_predictions_and_judge_provenance(tmp_path, point):
    manifest = [{"audio_id": "fixture", "pred_text": "", "voice_code_bench_revision": scorer.DATASET_REVISION}]
    cache = judge.VerifierCache(tmp_path / "cache.json")
    with patch.object(judge, "local_runtime", return_value=nullcontext()), \
         patch.object(judge, "post_request", side_effect=lambda request: response(request, False)):
        report = scorer.score_manifest(manifest, [point], cache=cache, cache_mode="live-fill")
    assert report["correct_entities"] == 0
    assert report["total_entities"] == 1
    assert report["complete_test_split"] is True
    assert report["verifier"]["id"] == judge.VERIFIER_ID
    assert report["verifier"]["runtime"]["revision"] == judge.MODEL_REVISION
    assert report["verifier"]["transcript_layout"] == "full_and_segments"


@pytest.mark.parametrize("upload_fails", [False, True])
def test_hf_worker_checkpoints_components_and_surfaces_upload_failures(upload_fails):
    from scripts import score_voice_code_bench_job as worker

    snapshots = []
    def upload(bucket, *, add):
        if upload_fails:
            raise RuntimeError("Upload failed")
        local, remote = add[0]
        snapshots.append(json.loads(local.read_text()))
        assert remote == "qwen27b-test/cache.json"
        assert bucket == "fixture-bucket"

    def score(*, cache_factory):
        cache = cache_factory("/results/qwen27b-test/cache.json")
        cache.put("first", {"fixture": 1})
        cache.put("second", {"fixture": 2})

    with patch.object(worker, "snapshot_download", return_value="fixture-package"), \
         patch.object(worker.subprocess, "run"), patch.object(worker, "HfApi") as api, \
         patch.object(worker.scorer, "main", side_effect=score), \
         patch.dict("os.environ", {"RESULTS_BUCKET": "fixture-bucket"}):
        api.return_value.batch_bucket_files.side_effect = upload
        if upload_fails:
            with pytest.raises(RuntimeError, match="Upload failed"):
                worker.main()
        else:
            worker.main()
            assert [len(snapshot["responses"]) for snapshot in snapshots] == [1, 2]


def test_rejects_wrong_returned_model(point):
    request = judge.build_request(point, "port 7326")
    raw = response(request)
    raw["model"] = "different-model"
    with pytest.raises(ValueError, match="Invalid Qwen response"):
        judge.parse_response(raw, request)


@pytest.mark.parametrize("failure", [None, "startup", "grading"])
def test_runtime_pins_weights_and_stops_on_success_or_failure(tmp_path, failure):
    server = MagicMock()
    server.poll.return_value = 1 if failure == "startup" else None
    health = MagicMock()
    health.__enter__.return_value.status = 200
    with patch("importlib.metadata.version", return_value="0.28.0"), \
         patch.object(judge.socket, "socket"), \
         patch.object(judge.subprocess, "Popen", return_value=server) as launch, \
         patch.object(judge.urllib.request, "urlopen", return_value=health):
        def run():
            with judge.local_runtime(tmp_path / "engine.log"):
                if failure == "grading":
                    raise ValueError("Grading failed")
        if failure:
            with pytest.raises((RuntimeError, ValueError)):
                run()
        else:
            run()
    command = launch.call_args.args[0]
    assert command[command.index("--revision") + 1] == judge.MODEL_REVISION
    assert command[command.index("--tokenizer-revision") + 1] == judge.MODEL_REVISION
    assert "--no-enable-log-requests" in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    if failure != "startup":
        server.terminate.assert_called_once()
        server.wait.assert_called_once_with(timeout=30)


def test_runtime_and_prompt_changes_invalidate_cache():
    original = judge.config_digest()
    with patch.dict(judge.RUNTIME, revision="different-revision"):
        assert judge.config_digest() != original
    with patch.object(judge, "SYSTEM_PROMPT", "different prompt"):
        assert judge.config_digest() != original
