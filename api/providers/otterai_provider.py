import io
import os
import time
from typing import Optional
from urllib.parse import urlparse

import librosa
import requests
import soundfile as sf

from . import APIProvider, PermanentError, register


@register("OtterAI")
class OtterAIProvider(APIProvider):
    def __init__(self):
        self.api_key = os.getenv("OTTERAI_API_KEY")
        self.api_url = os.getenv("OTTERAI_API_URL") or "https://research.otter.ai/v1/service/otter-speech"
        if not self.api_key:
            raise PermanentError("OTTERAI_API_KEY environment variable not set")
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
                "OTTERAI_API_URL must be an HTTPS URL without credentials, query, or fragment"
            )

    def _poll_long_form(self, job_id, headers):
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            response = requests.get(
                self.api_url, params={"job_id": job_id}, headers=headers,
                timeout=30, allow_redirects=False,
            )
            if response.status_code != 200:
                raise PermanentError(f"OtterAI job polling returned HTTP {response.status_code}")
            payload = response.json()
            status = payload.get("status") if isinstance(payload, dict) else None
            if status == "completed":
                return payload
            if status not in ("queued", "running"):
                raise PermanentError(f"OtterAI long-form job ended with status {status!r}")
            time.sleep(10)
        raise PermanentError("OtterAI long-form job did not finish within one hour")

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
                raise PermanentError("OtterAI requires an audio file")
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        wav = io.BytesIO()
        sf.write(wav, audio, 16000, format="WAV", subtype="PCM_16")

        long_form = os.getenv("OTTERAI_LONG_FORM") == "1"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"language": "en", "response_format": "json"}
        if long_form:
            data["long-form"] = "true"
        response = requests.post(
            self.api_url,
            headers=headers,
            files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
            data=data,
            timeout=300,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise PermanentError("OtterAI returned a redirect; check OTTERAI_API_URL")
        if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
            raise PermanentError(
                f"OtterAI returned HTTP {response.status_code}; check credentials, model, and request format"
            )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentError("OtterAI returned an invalid JSON response") from exc
        job_id = None
        if long_form and response.status_code == 202:
            job_id = payload.get("id") if isinstance(payload, dict) else None
            if not isinstance(job_id, str) or not job_id:
                raise PermanentError("OtterAI did not return a long-form job ID")
            payload = self._poll_long_form(job_id, headers)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise PermanentError("OtterAI response must contain a string 'text' field")
        if job_id:
            try:
                requests.delete(
                    self.api_url, params={"job_id": job_id}, headers=headers,
                    timeout=15, allow_redirects=False,
                )
            except requests.RequestException:
                pass  # The server expires unacknowledged results.
        return payload["text"]
