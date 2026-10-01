import os
import time
from collections import deque
from threading import Condition
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

DEFAULT_BASE_URL = "https://api.sprag.ai/v1"


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


RATE_LIMITER = _RequestRateLimiter("SPRAG_REQUESTS_PER_MINUTE", 2400)


@register("sprag")
class SpragProvider(APIProvider):
    def __init__(self):
        self.api_key = os.getenv("SPRAG_API_KEY")
        self.base_url = os.getenv("SPRAG_BASE_URL") or DEFAULT_BASE_URL

        if not self.api_key or self.api_key == "your_api_key":
            raise ValueError("SPRAG_API_KEY environment variable not set")

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if use_url or audio_file_path is None:
            raise PermanentError(
                "sprag provider requires local audio files (drop --use_url)"
            )

        data = {
            "model": model_variant,
            "temperature": 0.0,
        }
        if prompt:
            data["prompt"] = prompt

        RATE_LIMITER.acquire()
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={
                    "file": (
                        os.path.basename(audio_file_path),
                        audio_file,
                        "audio/wav",
                    )
                },
                data=data,
                timeout=600,
            )

        if response.status_code != 429 and 400 <= response.status_code < 500:
            raise PermanentError(f"HTTP {response.status_code}: {response.text[:500]}")

        response.raise_for_status()
        return response.json()["text"].strip()
