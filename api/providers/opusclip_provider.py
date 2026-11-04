import os
import time
import uuid
from typing import Optional
from urllib.parse import urlparse

import requests

from . import APIProvider, PermanentError, register

DEFAULT_API_URL = "https://platform.labs.opus.pro/api/v1"

# Languages each model accepts (ISO 639-1, as passed by the leaderboard scripts).
SUPPORTED_LANGUAGES = {"steno-v1-preview": {"en"}}

# A multipart request may carry at most 5 MiB in total. Leave room for the form
# fields; larger audio goes through a signed upload (POST /uploads) instead.
MULTIPART_AUDIO_LIMIT_BYTES = 5 * 1024 * 1024 - 64 * 1024

# The API asks for at least 5 s between polls of a job. Jobs fail server-side
# after 1,800 s (job_timeout, not charged), so stop waiting shortly after that.
POLL_INTERVAL_S = 5
JOB_DEADLINE_S = 1_860

# Attempts per HTTP call on 429, 5xx and network errors. A cold start after the
# model has been idle takes 1-1.5 minutes, but that happens inside the job, not
# in these calls.
MAX_HTTP_ATTEMPTS = 6
MAX_BACKOFF_S = 60

# Job failures that the same audio would hit again.
PERMANENT_JOB_ERRORS = {"audio_too_long", "invalid_media"}

# 4xx answers that no retry can fix: bad key, model not allowed for the key,
# no balance, a rejected request.
PERMANENT_STATUS = {400, 401, 402, 403, 404, 413, 422}
# 4xx answers that a fresh attempt (a new upload) can fix: upload_incomplete,
# upload_already_used, upload_expired. 409 idempotency_conflict cannot happen,
# since every submit replays the same body under its own key.
FRESH_ATTEMPT_STATUS = {409, 410}


class _TransientError(Exception):
    """An error worth a fresh attempt; run_eval.py retries the whole transcription."""


@register("opusclip")
class OpusClipProvider(APIProvider):
    """OpusClip API Platform speech-to-text (https://platform.labs.opus.pro/models/opusclip/steno-v1-preview).

    `opusclip/steno-v1-preview`: English only, up to 90 s of audio per request, no prompt,
    no word timestamps. Transcription runs as an async job: the audio is submitted
    to POST /audio/transcriptions (as a multipart `file`, or as a signed upload when
    the file is over 5 MiB), then GET /jobs/{id} is polled until the job succeeds
    or fails. A failed job is not charged.

    Env: OPUSCLIP_API_KEY; optional OPUSCLIP_API_URL (default https://platform.labs.opus.pro/api/v1).
    """

    def __init__(self):
        self.api_key = os.getenv("OPUSCLIP_API_KEY")
        self.api_url = (os.getenv("OPUSCLIP_API_URL") or DEFAULT_API_URL).rstrip("/")
        if not self.api_key:
            raise PermanentError("OPUSCLIP_API_KEY environment variable not set")
        url = urlparse(self.api_url)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise PermanentError("OPUSCLIP_API_URL must be an HTTPS URL without credentials, query, or fragment")
        self.auth = {"Authorization": f"Bearer {self.api_key}"}

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if model_variant not in SUPPORTED_LANGUAGES:
            raise PermanentError(
                f"Unknown model variant {model_variant!r}; use one of "
                + ", ".join(f"opusclip/{v}" for v in sorted(SUPPORTED_LANGUAGES))
            )
        language = language or "en"
        if language not in SUPPORTED_LANGUAGES[model_variant]:
            raise PermanentError(f"opusclip/{model_variant} does not support language {language!r}")
        if prompt is not None:
            raise PermanentError(f"opusclip/{model_variant} does not accept prompts")

        if use_url:
            audio_response = requests.get(sample["row"]["audio"][0]["src"], timeout=60)
            audio_response.raise_for_status()
            audio_bytes, filename = audio_response.content, "audio.wav"
        else:
            if not audio_file_path:
                raise PermanentError("opusclip provider requires an audio file")
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()
            filename = os.path.basename(audio_file_path)

        # One key per transcription: retries of the submit below replay the same
        # job instead of creating (and paying for) a second one.
        idempotency_key = f"oal-{uuid.uuid4()}"
        job = self._submit(f"opusclip/{model_variant}", language, audio_bytes, filename, idempotency_key)
        job = self._wait(job)

        if job["status"] == "succeeded":
            result = job.get("result") or {}
            text = result.get("text")
            if not isinstance(text, str):
                raise _TransientError(f"job {job['id']} succeeded without a text result")
            return text

        error = job.get("error") or {}
        code = error.get("code", job["status"])
        message = f"job {job['id']} {job['status']}: {code}: {error.get('message', '')}"
        if code in PERMANENT_JOB_ERRORS:
            raise PermanentError(message)
        raise _TransientError(message)

    def _submit(self, model: str, language: str, audio_bytes: bytes, filename: str, idempotency_key: str) -> dict:
        headers = {**self.auth, "Idempotency-Key": idempotency_key}
        url = f"{self.api_url}/audio/transcriptions"
        if len(audio_bytes) <= MULTIPART_AUDIO_LIMIT_BYTES:
            return self._call(
                "POST",
                url,
                headers=headers,
                data={"model": model, "language": language},
                files={"file": (filename, audio_bytes, "audio/wav")},
                timeout=120,
            )
        upload_id = self._upload(audio_bytes, filename)
        return self._call(
            "POST",
            url,
            headers=headers,
            json={"model": model, "language": language, "upload_id": upload_id},
            timeout=120,
        )

    def _upload(self, audio_bytes: bytes, filename: str) -> str:
        """Upload a file over the multipart limit to a signed URL; returns its single-use upload id."""
        upload = self._call(
            "POST",
            f"{self.api_url}/uploads",
            headers=self.auth,
            json={"filename": filename, "content_type": "audio/wav", "size_bytes": len(audio_bytes)},
            timeout=30,
        )
        try:
            put = requests.put(upload["upload_url"], headers=upload["headers"], data=audio_bytes, timeout=300)
        except requests.RequestException as e:
            raise _TransientError(f"upload PUT failed: {type(e).__name__}") from e
        if not put.ok:
            raise _TransientError(f"upload PUT returned HTTP {put.status_code}")
        return upload["id"]

    def _wait(self, job: dict) -> dict:
        deadline = time.monotonic() + JOB_DEADLINE_S
        while job["status"] not in ("succeeded", "failed", "canceled"):
            if time.monotonic() > deadline:
                raise _TransientError(f"job {job['id']} still {job['status']} after {JOB_DEADLINE_S} s")
            time.sleep(POLL_INTERVAL_S)
            job = self._call("GET", f"{self.api_url}/jobs/{job['id']}", headers=self.auth, timeout=30)
        return job

    def _call(self, method: str, url: str, **kwargs) -> dict:
        """One API call with bounded retries on 429, 5xx and network errors; returns the JSON body."""
        path = urlparse(url).path
        last = ""
        for attempt in range(MAX_HTTP_ATTEMPTS):
            backoff = min(2 ** (attempt + 1), MAX_BACKOFF_S)
            try:
                r = requests.request(method, url, allow_redirects=False, **kwargs)
            except requests.RequestException as e:
                last = type(e).__name__
                if attempt < MAX_HTTP_ATTEMPTS - 1:
                    time.sleep(backoff)
                continue
            if r.status_code < 300:
                try:
                    return r.json()
                except ValueError as e:
                    raise _TransientError(f"{method} {path}: invalid JSON response") from e
            if r.status_code < 400:
                # A host behind an access gate answers with a redirect to its login page.
                raise PermanentError(f"{method} {path} was redirected (HTTP {r.status_code}); check OPUSCLIP_API_URL")
            code, message = _error_of(r)
            last = f"HTTP {r.status_code} {code}: {message}"
            if r.status_code in PERMANENT_STATUS:
                raise PermanentError(f"{method} {path}: {last}")
            if r.status_code in FRESH_ATTEMPT_STATUS:
                raise _TransientError(f"{method} {path}: {last}")
            # 429, other 4xx (e.g. 408) and every 5xx: wait and retry.
            if r.status_code == 429:
                try:
                    backoff = min(max(float(r.headers.get("retry-after", backoff)), 1.0), MAX_BACKOFF_S)
                except ValueError:
                    pass
            if attempt < MAX_HTTP_ATTEMPTS - 1:
                time.sleep(backoff)
        raise _TransientError(f"{method} {path} failed after {MAX_HTTP_ATTEMPTS} attempts: {last}")


def _error_of(r: requests.Response) -> tuple[str, str]:
    """The API's error code and message, from {"error": {"type", "code", "message"}}."""
    try:
        error = r.json().get("error") or {}
        return str(error.get("code", "")), str(error.get("message", ""))[:300]
    except (ValueError, AttributeError):
        return "", ""
