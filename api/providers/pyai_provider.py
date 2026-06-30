import os
import time
from io import BytesIO
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

PYAI_API_BASE = "https://api.pyai.com"


@register("pyai")
class PyAIProvider(APIProvider):
    """PyAI Hear — streaming STT API for voice agents.

    API docs: https://api.pyai.com/docs
    Sign up for 1,000,000 free minutes/month: https://api.pyai.com

    Set env var: PYAI_API_KEY
    Model variants:
      pyai/hear-v4   — production model (streaming-optimised FastConformer-TDT)
    """

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        **kwargs,
    ) -> str:
        api_key = os.getenv("PYAI_API_KEY")
        if not api_key:
            raise PermanentError("PYAI_API_KEY environment variable not set")

        headers = {"Authorization": f"Bearer {api_key}"}

        if use_url:
            audio_url = sample["row"]["audio"][0]["src"]
            resp = requests.get(audio_url, timeout=30)
            if not resp.ok:
                raise PermanentError(f"Failed to fetch audio URL: {resp.status_code}")
            audio_bytes = BytesIO(resp.content)
            filename = "audio.wav"
        else:
            audio_bytes = open(audio_file_path, "rb")
            filename = os.path.basename(audio_file_path)

        try:
            response = requests.post(
                f"{PYAI_API_BASE}/v1/audio/transcriptions",
                headers=headers,
                files={"file": (filename, audio_bytes, "audio/wav")},
                data={
                    "model": model_variant,
                    "language": language,
                    "response_format": "json",
                },
                timeout=120,
            )
        finally:
            if not use_url:
                audio_bytes.close()

        if response.status_code == 401:
            raise PermanentError(f"PyAI auth error: {response.text}")
        if response.status_code == 429:
            # Rate limited — let the retry loop handle it
            raise Exception(f"PyAI rate limit (429): {response.text}")
        if not response.ok:
            raise Exception(f"PyAI API error {response.status_code}: {response.text}")

        result = response.json()
        return result.get("text", "")
