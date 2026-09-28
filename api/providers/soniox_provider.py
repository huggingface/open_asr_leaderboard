import os
from collections import deque
from threading import Condition
import time
from typing import Optional

import requests

from . import APIProvider, PermanentError, register


DEFAULT_MODEL = "stt-async-v5"
DEFAULT_BASE_URL = "https://api.soniox.com/v1"
_COMPLETED_STATUSES = {"completed", "done", "succeeded", "success"}
_ERROR_STATUSES = {"error", "failed", "canceled", "cancelled"}


class _RequestRateLimiter:
    """Process-wide sliding-window limiter shared by all worker threads."""

    def __init__(self, env_name: str, default_requests_per_minute: int):
        configured = int(os.getenv(env_name, str(default_requests_per_minute)))
        self.limit = max(1, configured)
        self._condition = Condition()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        with self._condition:
            while True:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.limit:
                    self._timestamps.append(now)
                    return
                wait_seconds = max(0.05, self._timestamps[0] + 60.0 - now)
                self._condition.wait(timeout=wait_seconds)


# Keep request bursts bounded below the limits observed during the public
# benchmark. Exact RPM caps are project- and organization-specific in Soniox.
_FILE_RATE_LIMITER = _RequestRateLimiter(
    "SONIOX_FILE_REQUESTS_PER_MINUTE", 240
)
_TRANSCRIPTION_RATE_LIMITER = _RequestRateLimiter(
    "SONIOX_TRANSCRIPTION_REQUESTS_PER_MINUTE", 240
)


def _response_error(response: requests.Response) -> str:
    """Return a bounded API error without including request headers."""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error_type = str(payload.get("error_type") or "").strip()
        message = str(
            payload.get("message") or payload.get("error_message") or ""
        ).strip()
        if error_type and message:
            return f"{error_type}: {message}"[:1000]
        if error_type or message:
            return (error_type or message)[:1000]

    return (response.text or f"HTTP {response.status_code}")[:1000]


@register("soniox")
class SonioxProvider(APIProvider):
    def __init__(self):
        api_key = os.getenv("SONIOX_API_KEY")
        if not api_key:
            raise ValueError("SONIOX_API_KEY environment variable not set")

        self.base_url = os.getenv("SONIOX_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.session = requests.Session()

    @staticmethod
    def _raise_for_status(response: requests.Response, operation: str) -> None:
        if response.ok:
            return

        message = f"Soniox {operation} failed ({response.status_code}): {_response_error(response)}"
        if response.status_code in {400, 401, 402, 403, 404}:
            raise PermanentError(message)
        raise RuntimeError(message)

    def _delete(self, resource: str, resource_id: Optional[str]) -> None:
        if not resource_id:
            return
        try:
            limiter = (
                _FILE_RATE_LIMITER
                if resource == "files"
                else _TRANSCRIPTION_RATE_LIMITER
            )
            limiter.acquire()
            response = self.session.delete(
                f"{self.base_url}/{resource}/{resource_id}",
                headers=self.headers,
                timeout=(10, 30),
            )
            if response.status_code not in {200, 204, 404}:
                print(
                    f"Warning: Soniox cleanup for {resource} returned "
                    f"HTTP {response.status_code}."
                )
        except requests.RequestException as exc:
            print(f"Warning: Soniox cleanup for {resource} failed: {exc}")

    def _get_with_retry(self, url: str, operation: str) -> requests.Response:
        delay = 1.0
        for attempt in range(10):
            _TRANSCRIPTION_RATE_LIMITER.acquire()
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(10, 60),
            )
            if response.status_code != 429 and response.status_code < 500:
                self._raise_for_status(response, operation)
                return response
            if attempt == 9:
                self._raise_for_status(response, operation)
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else delay
            except ValueError:
                wait_seconds = delay
            time.sleep(max(1.0, min(wait_seconds, 60.0)))
            delay = min(delay * 2, 30.0)
        raise RuntimeError(f"Soniox {operation} failed after retries")

    def _post_with_retry(
        self,
        url: str,
        operation: str,
        limiter: _RequestRateLimiter,
        **kwargs,
    ) -> requests.Response:
        delay = 1.0
        for attempt in range(10):
            limiter.acquire()
            response = self.session.post(url, headers=self.headers, **kwargs)
            if response.status_code != 429 and response.status_code < 500:
                self._raise_for_status(response, operation)
                return response
            if attempt == 9:
                self._raise_for_status(response, operation)
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else delay
            except ValueError:
                wait_seconds = delay
            time.sleep(max(1.0, min(wait_seconds, 60.0)))
            delay = min(delay * 2, 30.0)
        raise RuntimeError(f"Soniox {operation} failed after retries")

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        del prompt  # Soniox context is intentionally not varied between datasets.

        model = (model_variant or DEFAULT_MODEL).strip()
        file_id = None
        transcription_id = None

        try:
            payload = {
                "model": model,
                "language_hints": [language],
                "language_hints_strict": True,
                "enable_speaker_diarization": False,
                "enable_language_identification": False,
            }

            if use_url:
                try:
                    payload["audio_url"] = sample["row"]["audio"][0]["src"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise PermanentError(
                        "Dataset row does not contain a usable public audio URL"
                    ) from exc
            else:
                if not audio_file_path:
                    raise PermanentError("Soniox local-file evaluation requires an audio path")
                with open(audio_file_path, "rb") as audio_file:
                    response = self._post_with_retry(
                        f"{self.base_url}/files",
                        "file upload",
                        _FILE_RATE_LIMITER,
                        files={
                            "file": (
                                os.path.basename(audio_file_path),
                                audio_file,
                                "audio/wav",
                            )
                        },
                        timeout=(10, 180),
                    )
                file_id = str(response.json().get("id") or "").strip()
                if not file_id:
                    raise RuntimeError("Soniox file upload response did not include an id")
                payload["file_id"] = file_id

            response = self._post_with_retry(
                f"{self.base_url}/transcriptions",
                "transcription creation",
                _TRANSCRIPTION_RATE_LIMITER,
                json=payload,
                timeout=(10, 60),
            )
            transcription_id = str(response.json().get("id") or "").strip()
            if not transcription_id:
                raise RuntimeError(
                    "Soniox transcription creation response did not include an id"
                )

            timeout_seconds = float(os.getenv("SONIOX_POLL_TIMEOUT_SECONDS", "900"))
            deadline = time.monotonic() + timeout_seconds
            poll_delay = 5.0
            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise TimeoutError("Soniox transcription polling timed out")

                # Async jobs are never complete at creation time. Waiting before
                # the first status check avoids one guaranteed no-op request per
                # sample and preserves the project-specific Async RPM budget.
                time.sleep(min(poll_delay, remaining_seconds))

                status_response = self._get_with_retry(
                    f"{self.base_url}/transcriptions/{transcription_id}",
                    "transcription polling",
                )
                status_payload = status_response.json()
                status = str(status_payload.get("status") or "").lower()
                if status in _COMPLETED_STATUSES:
                    break
                if status in _ERROR_STATUSES:
                    error_type = str(status_payload.get("error_type") or "").strip()
                    error_message = str(
                        status_payload.get("error_message") or "Soniox transcription failed"
                    ).strip()
                    detail = f"{error_type}: {error_message}" if error_type else error_message
                    raise PermanentError(detail)

                poll_delay = min(poll_delay * 1.5, 10.0)

            transcript_response = self._get_with_retry(
                f"{self.base_url}/transcriptions/{transcription_id}/transcript",
                "transcript retrieval",
            )
            return str(transcript_response.json().get("text") or "").strip()
        finally:
            # Transcriptions reference files, so remove them first.
            self._delete("transcriptions", transcription_id)
            self._delete("files", file_id)
