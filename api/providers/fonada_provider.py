import os
from typing import Optional
from urllib.parse import urlparse

import requests

from . import APIProvider, PermanentError, register

# Languages Kapture v3 transcribes (ISO 639-1 codes, as passed by the leaderboard scripts).
SUPPORTED_LANGUAGES = {"en", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te"}


@register("fonada")
class FonadaProvider(APIProvider):
    """FonadaLabs Kapture v3 (https://fonadalabs.ai/docs/speech-to-text/kapture-v3).

    `fonada/kapture-v3`: multipart POST of the audio file to /v3/asr/transcribe with the
    language code; long audio is split and joined server-side. No prompt is used.
    """

    def __init__(self):
        self.api_key = os.getenv("FONADA_API_KEY")
        self.api_url = os.getenv("FONADA_API_URL") or "https://api.fonada.ai/v3/asr/transcribe"
        if not self.api_key:
            raise PermanentError("FONADA_API_KEY environment variable not set")
        url = urlparse(self.api_url)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise PermanentError("FONADA_API_URL must be an HTTPS URL without credentials, query, or fragment")

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if model_variant != "kapture-v3":
            raise PermanentError(f"Unknown model variant {model_variant!r}; use fonada/kapture-v3")
        if language not in SUPPORTED_LANGUAGES:
            raise PermanentError(f"Kapture v3 does not support language {language!r}")
        if prompt is not None:
            raise PermanentError("This adapter does not accept prompts")

        if use_url:
            audio_response = requests.get(sample["row"]["audio"][0]["src"], timeout=60)
            audio_response.raise_for_status()
            audio_bytes, name = audio_response.content, "audio.wav"
        else:
            if not audio_file_path:
                raise PermanentError("Kapture v3 requires an audio file")
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            name = os.path.basename(audio_file_path)

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": (name, audio_bytes, "application/octet-stream")},
            data={"language_id": language},
            timeout=300,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise PermanentError("Fonada returned a redirect; check FONADA_API_URL")
        if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
            raise PermanentError(
                f"Fonada returned HTTP {response.status_code}; check credentials and request format"
            )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentError("Fonada returned an invalid JSON response") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise PermanentError("Fonada response must contain a string 'text' field")
        return payload["text"]
