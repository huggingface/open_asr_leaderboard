import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

SUPPORTED_LANGUAGES = {"en", "de"}


@register("sophea")
class SopheaProvider(APIProvider):
    """Sophea ASR (KIEFER SA). Model variants: "asr-k1" (English and German).
    Env: SOPHEA_API_KEY, SOPHEA_API_URL (default https://api.sophea.ai)."""

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        language = (language or "en").strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            raise PermanentError(
                f"Sophea {model_variant} supports {sorted(SUPPORTED_LANGUAGES)}, got '{language}'."
            )
        base = os.getenv("SOPHEA_API_URL", "https://api.sophea.ai").rstrip("/")
        headers = {"X-API-Key": os.environ["SOPHEA_API_KEY"]}
        data = {"model": model_variant, "language": language}
        if use_url:
            audio_bytes = requests.get(sample["row"]["audio"][0]["src"], timeout=60).content
            files = {"file": ("audio", audio_bytes)}
        else:
            with open(audio_file_path, "rb") as f:
                files = {"file": (os.path.basename(audio_file_path), f.read())}
        r = requests.post(f"{base}/v1/transcribe", headers=headers, files=files, data=data, timeout=300)
        r.raise_for_status()
        return r.json()["text"]
