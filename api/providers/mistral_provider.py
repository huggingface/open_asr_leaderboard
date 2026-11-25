import os
from typing import Optional
from urllib.parse import urlparse

import requests

from . import APIProvider, PermanentError, register


DEFAULT_API_URL = "https://api.mistral.ai/v1/audio/transcriptions"
SUPPORTED_MODELS = {
    "voxtral-mini-transcribe-3",
    "voxtral-small-transcribe-3",
}


@register("mistral")
class MistralProvider(APIProvider):
    """Mistral Voxtral v3 transcription API.

    The English short-form submission uses forced English, deterministic decoding,
    and verbatim mode for every dataset.

    Env: MISTRAL_API_KEY; optional MISTRAL_API_URL.
    """

    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.api_url = (os.getenv("MISTRAL_API_URL") or DEFAULT_API_URL).rstrip("/")
        if not self.api_key:
            raise PermanentError("MISTRAL_API_KEY environment variable not set")
        url = urlparse(self.api_url)
        if (
            url.scheme != "https"
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise PermanentError(
                "MISTRAL_API_URL must be an HTTPS URL without credentials, query, or fragment"
            )

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if model_variant not in SUPPORTED_MODELS:
            supported = ", ".join(f"mistral/{model}" for model in sorted(SUPPORTED_MODELS))
            raise PermanentError(
                f"Unknown model variant {model_variant!r}; use one of {supported}"
            )
        if language != "en":
            raise PermanentError(
                "This submission is configured for the English short-form benchmark only"
            )
        if prompt is not None:
            raise PermanentError("The Mistral submission does not use a prompt")

        if use_url:
            audio_response = requests.get(sample["row"]["audio"][0]["src"], timeout=60)
            audio_response.raise_for_status()
            audio_bytes, filename = audio_response.content, "audio.wav"
        else:
            if not audio_file_path:
                raise PermanentError("Mistral provider requires an audio file")
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            filename = os.path.basename(audio_file_path)

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={
                "model": model_variant,
                "language": "en",
                "temperature": "0",
                "verbatim": "true",
            },
            timeout=600,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise PermanentError("Mistral returned a redirect; check MISTRAL_API_URL")
        if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
            raise PermanentError(
                f"Mistral returned HTTP {response.status_code}; check credentials and request format"
            )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentError("Mistral returned an invalid JSON response") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise PermanentError("Mistral response must contain a string 'text' field")
        return payload["text"].strip()
