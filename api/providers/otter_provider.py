import io
import os
from typing import Optional
from urllib.parse import urlparse

import librosa
import requests
import soundfile as sf

from . import APIProvider, PermanentError, register


@register("OtterAI")
class OtterProvider(APIProvider):
    def __init__(self):
        self.api_key = os.getenv("OTTER_API_KEY")
        self.api_url = os.getenv("OTTER_API_URL") or "https://research.otter.ai/v1/service/otter-speech"
        if not self.api_key:
            raise PermanentError("OTTER_API_KEY environment variable not set")
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
                "OTTER_API_URL must be an HTTPS URL without credentials, query, or fragment"
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
        if language != "en":
            raise PermanentError("This adapter accepts English only")
        if prompt is not None:
            raise PermanentError("This adapter does not accept prompts")

        if use_url:
            audio_response = requests.get(sample["row"]["audio"][0]["src"], timeout=60)
            audio_response.raise_for_status()
            audio_bytes = audio_response.content
        else:
            if not audio_file_path:
                raise PermanentError("Otter requires an audio file")
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        wav = io.BytesIO()
        sf.write(wav, audio, 16000, format="WAV", subtype="PCM_16")

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
            data={"language": "en", "response_format": "json"},
            timeout=300,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise PermanentError("Otter returned a redirect; check OTTER_API_URL")
        if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
            raise PermanentError(
                f"Otter returned HTTP {response.status_code}; check credentials, model, and request format"
            )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentError("Otter returned an invalid JSON response") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise PermanentError("Otter response must contain a string 'text' field")
        return payload["text"]
