"""VoiceCodeBench judging with a pinned, locally executed open-weight Qwen model."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from threading import RLock
import time
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator, ValidationError
from tqdm import tqdm

VERIFIER_ID = "qwen3_8_27b_full_and_segments_v1"
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "8c0b3d30f3ae96f5c7f774de704bf0ada83e422a"
RUNTIME_IMAGE = "vllm/vllm-openai:v0.28.0-cu129"
RUNTIME = {"model": MODEL, "revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
           "image": RUNTIME_IMAGE, "dtype": "bfloat16", "max_model_len": 32768,
           "max_num_seqs": 16, "gpu_memory_utilization": 0.9,
           "reasoning_parser": "qwen3", "generation_config": "vllm"}
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT_PATH = Path(__file__).with_name("voice_code_bench_prompt.txt")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def stable_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def config_digest():
    # Bind request construction, validation and pinned runtime as well as
    # the prompt. Old judge caches must never become new judge observations.
    return stable_digest({
        "id": VERIFIER_ID,
        "runtime": RUNTIME,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system_prompt": SYSTEM_PROMPT,
    })


def build_request(datapoint, transcript):
    segments = [transcript[i:i + 256] for i in range(0, len(transcript), 256)] or [""]
    targets = [
        {"target_index": i, **{key: entity[key] for key in
         ("id", "type", "role", "acoustic", "canonical", "alternative_spellings")}}
        for i, entity in enumerate(datapoint["entities"])
    ]
    data = {"datapoint_id": datapoint["audio_id"], "target_entities": targets, "transcript": transcript}
    data["transcript_segments"] = [{"segment_index": i, "text": text} for i, text in enumerate(segments)]
    entries = {}
    for target in targets:
        properties = {
            "target_index": {"type": "integer", "enum": [target["target_index"]]},
            "type": {"type": "string", "enum": [target["type"]]},
            "canonical": {"type": "string", "enum": [target["canonical"]]},
            "evidence_start_segment": {"type": "integer", "enum": list(range(-1, len(segments)))},
            "evidence_end_segment": {"type": "integer", "enum": list(range(-1, len(segments)))},
            "reason": {"type": "string"},
            "present": {"type": "boolean"},
        }
        entries[str(target["target_index"])] = {
            "type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties),
        }
    schema = {
        "type": "object", "additionalProperties": False, "required": ["entities"],
        "properties": {"entities": {"type": "object", "additionalProperties": False,
                                    "properties": entries, "required": list(entries)}},
    }
    return {
        "model": MODEL, "temperature": 0,
        "max_tokens": 16384,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "fixed_identity_verdicts", "strict": True, "schema": schema}},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
        "thinking_token_budget": 8192,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key in judge data.")
        result[key] = value
    return result


def parse_response(response, request):
    """Validate network/cache data before turning citations into exact evidence."""
    try:
        if response["model"] != MODEL:
            raise ValueError("Unexpected judge model.")
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete Qwen completion.")
        parsed = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
        Draft202012Validator(request["response_format"]["json_schema"]["schema"]).validate(parsed)
        data = json.loads(request["messages"][1]["content"])
        segments = [segment["text"] for segment in data["transcript_segments"]]
        matches = []
        for i in range(len(data["target_entities"])):
            entry = parsed["entities"][str(i)]
            start, end = entry["evidence_start_segment"], entry["evidence_end_segment"]
            if any(type(value) is not int for value in (entry["target_index"], start, end)):
                raise ValueError("Qwen indices must be integers.")
            if start == end == -1 and entry["present"] is False:
                evidence = ""
            elif 0 <= start <= end < len(segments):
                evidence = "".join(segments[start:end + 1])
            else:
                raise ValueError("Invalid Qwen citation range.")
            if not entry["reason"].strip() or (entry["present"] and not evidence.strip()):
                raise ValueError("Qwen verdict needs a reason and positives need evidence.")
            matches.append({**{key: entry[key] for key in
                              ("target_index", "type", "canonical", "reason", "present")},
                            "evidence": evidence})
        return matches
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        # Do not include transcripts, model output or target values in logs.
        raise ValueError("Invalid Qwen response; scoring stopped.") from None


class VerifierCache:
    """Thread-safe, atomic response cache with an explicit new format."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = RLock()
        self.header = {"schema_version": 2, "verifier_id": VERIFIER_ID, "config_digest": config_digest()}
        self.responses = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
            if set(data) != {*self.header, "responses"} or any(data[key] != value for key, value in self.header.items()):
                raise ValueError("Incompatible verifier cache. Use a new Qwen cache path; old judgments cannot be reused.")
            if not isinstance(data["responses"], dict):
                raise ValueError("Invalid Qwen response cache.")
            if any(not isinstance(response, dict) for response in data["responses"].values()):
                raise ValueError("Invalid cached Qwen response.")
            self.responses = data["responses"]

    def get(self, digest):
        with self.lock:
            return self.responses.get(digest)

    def put(self, digest, response):
        with self.lock:
            if digest in self.responses:
                raise ValueError("A Qwen response is already recorded for this request.")
            self.responses[digest] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump({**self.header, "responses": self.responses}, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    temporary.unlink()
                    raise
            try:
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            # The HF worker overrides this hook; uploads share the write lock.
            self.checkpoint()

    def checkpoint(self):
        """Local cache is already durable; the HF worker also publishes it."""


def post_request(request):
    req = urllib.request.Request(ENDPOINT, data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response, object_pairs_hook=unique_object)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Qwen judge HTTP {exc.code}; no automatic retry was issued.") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Qwen judge request failed; no automatic retry was issued.") from None


def request_digest(request):
    return stable_digest({"runtime": RUNTIME, "request": request})


@contextmanager
def local_runtime(log_path):
    """Own one pinned vLLM server for the grading run; replay needs no server."""
    from importlib.metadata import version

    if version("vllm") != "0.28.0":
        raise RuntimeError(f"Use the tested grading image: {RUNTIME_IMAGE}.")
    # Refuse to use an unrelated server already listening on our endpoint.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8000))
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", MODEL, "--revision", MODEL_REVISION,
               "--tokenizer-revision", MODEL_REVISION, "--served-model-name", MODEL,
               "--dtype", "bfloat16", "--max-model-len", "32768", "--max-num-seqs", "16",
               "--gpu-memory-utilization", "0.9", "--reasoning-parser", "qwen3",
               "--generation-config", "vllm", "--language-model-only",
               "--no-enable-log-requests", "--host", "127.0.0.1", "--port", "8000"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 1200
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Qwen runtime exited before readiness; see {log_path}.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as health:
                    if health.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                # Only health checks repeat. An inference request is sent once.
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Qwen runtime readiness deadline exceeded; see {log_path}.")
            time.sleep(2)
        yield
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def verify_entity_matches(datapoint, transcript, *, cache, cache_mode):
    if cache_mode not in ("replay", "live-fill"):
        raise ValueError("Unknown verifier cache mode.")
    request = build_request(datapoint, transcript)
    digest = request_digest(request)
    response = cache.get(digest)
    if response is None:
        if cache_mode == "replay":
            raise ValueError("Qwen replay cache miss; no request was issued.")
        response = post_request(request)
        matches = parse_response(response, request)
        cache.put(digest, response)
        return matches
    return parse_response(response, request)


def verify_manifest(manifest, targets, *, cache, cache_mode, workers):
    """Own the local runtime, bound requests, and preserve manifest order."""
    if workers < 1:
        raise ValueError("Verifier workers must be positive.")
    if cache_mode not in ("replay", "live-fill"):
        raise ValueError("Unknown verifier cache mode.")
    missing = False
    for row in manifest:
        request = build_request(targets[row["audio_id"]], row["pred_text"])
        cached = cache.get(request_digest(request))
        if cached is None:
            if cache_mode == "replay":
                raise ValueError("Qwen replay cache miss; no request was issued.")
            missing = True
        else:
            parse_response(cached, request)
    runtime = local_runtime(cache.path.with_suffix(".engine.log")) if missing else nullcontext()
    matches = [None] * len(manifest)
    with runtime, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify_entity_matches, targets[row["audio_id"]], row["pred_text"],
                                   cache=cache, cache_mode=cache_mode): i for i, row in enumerate(manifest)}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying entities", unit="recording"):
                matches[futures[future]] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return matches
