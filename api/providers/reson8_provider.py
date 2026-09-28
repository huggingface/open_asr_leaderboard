import os
import time
import threading
from typing import Optional

import requests

from . import APIProvider, register, PermanentError


AUTH_URL = "https://api.reson8.dev/v1/auth/token"
TRANSCRIBE_URL = "https://api.reson8.dev/v1/speech-to-text/prerecorded"
TOKEN_REFRESH_MARGIN_S = 30
MIN_AUDIO_DURATION_S = 0.16

_token_lock = threading.Lock()
_token_cache: dict = {"access_token": None, "expires_at": 0.0}


def _get_access_token(api_key: str) -> str:
    with _token_lock:
        cached = _token_cache["access_token"]
        if cached and _token_cache["expires_at"] - time.time() > TOKEN_REFRESH_MARGIN_S:
            return cached

        response = requests.post(
            AUTH_URL,
            headers={"Authorization": f"ApiKey {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + float(data.get("expires_in", 600))
        return _token_cache["access_token"]


@register("reson8")
class Reson8Provider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        api_key = os.getenv("RESON8_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise PermanentError("RESON8_API_KEY is not set (or still the placeholder)")

        if use_url:
            audio_duration = sample["row"]["audio_length_s"]
            if audio_duration < MIN_AUDIO_DURATION_S:
                return "."
            audio_url = sample["row"]["audio"][0]["src"]
            resp = requests.get(audio_url, timeout=60)
            resp.raise_for_status()
            audio_bytes = resp.content
        else:
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )
            if audio_duration < MIN_AUDIO_DURATION_S:
                return "."
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()

        params = {"encoding": "auto", "model": model_variant}
        if language:
            params["language"] = language

        token = _get_access_token(api_key)
        response = requests.post(
            TRANSCRIBE_URL,
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
            data=audio_bytes,
            timeout=300,
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()
