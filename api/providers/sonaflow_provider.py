"""Sonaflow Inc. Bessel Consensus3 r1 English transcription API provider.

No sample metadata, reference, prompt, or dataset name is sent to the API.
Once a job might have been accepted, all recovery uses the SAME request ID.
Failures are PermanentError so the runner cannot silently start another job.
"""
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import requests

from . import APIProvider, PermanentError, register

VARIANT = "bessel-consensus3-r1"
MODEL_ID = "Sonaflow/bessel-consensus3-r1"
PROFILE_SHA256 = "5c0bcaad39f97b37f6dd073e26dc65b41f6a3919af8befc6580cbe31b30d80c4"
RUNTIME_SHA256 = "84ad4385c75b5260bc8ba58eacb328112e2d34805a5bca30d224e88738d83899"


@register("Sonaflow")
class BesselProvider(APIProvider):
    def transcribe(self, model_variant, audio_file_path, sample, use_url=False,
                   language="en", prompt=None):
        identity = uuid.uuid4().hex
        try:
            return self._transcribe(identity, model_variant, audio_file_path, use_url, language, prompt)
        except PermanentError:
            raise
        except Exception as exc:
            # Never let the upstream generic retry loop mint another paid job.
            raise PermanentError(f"Bessel client {type(exc).__name__}; preserve request {identity}; no replay") from None

    def _transcribe(self, identity, model_variant, audio_file_path, use_url, language, prompt):
        if model_variant != VARIANT or use_url or not audio_file_path or language != "en" or prompt:
            raise PermanentError("Bessel Consensus3 r1 requires its versioned English model, a local WAV/FLAC and no prompt")
        url = os.environ.get("BESSEL_API_URL", "").rstrip("/")
        key = os.environ.get("BESSEL_API_KEY", "")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ("", "/") or len(key) < 32):
            raise PermanentError("Configure an HTTPS BESSEL_API_URL and private BESSEL_API_KEY")
        audio = Path(audio_file_path).read_bytes()
        if not audio or len(audio) > 24 * 1024 * 1024:
            raise PermanentError("Bessel accepts audio files up to 24 MiB")
        # A unique invocation ID avoids cross-recording and cross-run result caches.
        headers = {"Authorization": "Bearer " + key, "Idempotency-Key": identity}
        deadline = time.monotonic() + 7200
        with requests.Session() as session:
            session.trust_env = False
            session.headers.update(headers)

            def exchange(method, suffix, **kwargs):
                try:
                    r = session.request(method, url + suffix, timeout=(15, 60),
                                        allow_redirects=False, **kwargs)
                except requests.RequestException:
                    return None
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    return None
                if r.status_code == 404 and method == "GET":
                    return {"status": "not_found"}
                if r.status_code not in (200, 202):
                    raise PermanentError(f"Bessel rejected HTTP {r.status_code}; request {identity}; no replay")
                try:
                    value = r.json()
                    return value if isinstance(value, dict) else None
                except ValueError:
                    return None

            # Service startup or transient outages can exceed one HTTP timeout.
            # Retrying this metadata GET submits no audio and creates no job.
            seal = None
            for attempt in range(10):
                seal = exchange("GET", "/v1/evaluation-contract")
                if seal is not None:
                    break
                time.sleep(min(2 ** attempt, 10))
            if not seal or seal.get("profile_sha256") != PROFILE_SHA256 or seal.get("inference_runtime_sha256") != RUNTIME_SHA256 or seal.get("upload_audio_storage") != "memory_only" or seal.get("model") != MODEL_ID:
                raise PermanentError("Bessel Consensus3 r1 endpoint contract is unavailable or mismatched; no audio submitted")
            accepted = False
            value = None
            for attempt in range(6):
                # GET before a repeated POST resolves lost responses without replay.
                if attempt:
                    value = exchange("GET", "/v1/jobs/" + identity)
                    if value and value.get("status") != "not_found":
                        accepted = True
                        break
                value = exchange("POST", "/v1/jobs",
                    files={"file": ("audio.wav", audio, "audio/wav")},
                    data={"model": MODEL_ID, "language": "en"})
                if value:
                    accepted = True
                    break
                time.sleep(min(2 ** attempt, 10))
            del audio
            if not accepted:
                raise PermanentError(f"Bessel submission unresolved; preserve request {identity}; do not rerun")
            while time.monotonic() < deadline:
                if value:
                    if value.get("id") != identity:
                        raise PermanentError("Bessel returned an inconsistent job identity")
                    status = value.get("status")
                    if status == "complete":
                        result = value.get("result", {})
                        if (not isinstance(result, dict) or result.get("model") != MODEL_ID or result.get("profile_sha256") != PROFILE_SHA256
                                or result.get("runtime_sha256") != RUNTIME_SHA256 or not isinstance(result.get("text"), str)):
                            raise PermanentError("Bessel Consensus3 r1 result provenance mismatch")
                        # Preserve the API's normalized text and empty transcripts exactly.
                        return result["text"]
                    if status not in ("queued", "running"):
                        raise PermanentError(f"Bessel job {status}; preserve request {identity}; no replay")
                time.sleep(2)
                value = exchange("GET", "/v1/jobs/" + identity)
            raise PermanentError(f"Bessel polling timed out; preserve request {identity}; resume GET only")
