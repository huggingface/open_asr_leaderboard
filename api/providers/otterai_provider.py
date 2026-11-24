import io
import os
import time
import uuid
from typing import Optional
from urllib.parse import urlparse

import librosa
import requests
import soundfile as sf

from . import APIProvider, PermanentError, register

POLL_INTERVAL_S = 10
JOB_TIMEOUT_S = 3600


@register("OtterAI")
class OtterAIProvider(APIProvider):
    def __init__(self):
        self.api_key = os.getenv("OTTERAI_API_KEY")
        self.api_url = os.getenv("OTTERAI_API_URL") or "https://research.otter.ai/v1/service/otter-speech"
        self.long_form = os.getenv("OTTERAI_LONG_FORM", "0") == "1"
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

    @staticmethod
    def _check_response(response):
        if 300 <= response.status_code < 400:
            raise PermanentError("OtterAI returned a redirect; check OTTERAI_API_URL")
        if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
            raise PermanentError(
                f"OtterAI returned HTTP {response.status_code}; check credentials, model, and request format"
            )
        response.raise_for_status()

    @staticmethod
    def _payload(response):
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentError("OtterAI returned an invalid JSON response") from exc
        if not isinstance(payload, dict):
            raise PermanentError("OtterAI response must contain a JSON object")
        return payload

    @staticmethod
    def _text(payload):
        if not isinstance(payload.get("text"), str):
            raise PermanentError("OtterAI response must contain a string 'text' field")
        return payload["text"]

    def _submit(self, wav, headers, data):
        attempts = 3 if self.long_form else 1
        for attempt in range(attempts):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files={"file": ("audio.wav", wav, "audio/wav")},
                    data=data,
                    timeout=(30, 60) if self.long_form else 300,
                    allow_redirects=False,
                )
                self._check_response(response)
                return response
            except requests.RequestException:
                if attempt + 1 == attempts:
                    raise
                time.sleep(POLL_INTERVAL_S)

    def _poll(self, job_id, headers):
        params = {"job_id": job_id}
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                response = requests.get(
                    self.api_url,
                    params=params,
                    headers=headers,
                    timeout=min(30, remaining),
                    allow_redirects=False,
                )
                if response.status_code == 404:
                    raise PermanentError("OtterAI long-form job is missing or expired")
                self._check_response(response)
            except requests.RequestException:
                # Keep polling the accepted job after a transient error.
                pass
            else:
                payload = self._payload(response)
                status = payload.get("status")
                if status == "completed":
                    text = self._text(payload)
                    try:
                        requests.delete(
                            self.api_url,
                            params=params,
                            headers=headers,
                            timeout=15,
                            allow_redirects=False,
                        )
                    except requests.RequestException:
                        # The server also expires unacknowledged results.
                        pass
                    return text
                if status == "failed":
                    raise PermanentError("OtterAI long-form transcription failed")
                if status not in ("queued", "running"):
                    raise PermanentError("OtterAI returned an unknown long-form job status")
            time.sleep(max(0, min(POLL_INTERVAL_S, deadline - time.monotonic())))
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

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"language": "en", "response_format": "json"}
        if self.long_form:
            headers["Idempotency-Key"] = uuid.uuid4().hex
            data["long-form"] = "true"
        response = self._submit(wav.getvalue(), headers, data)
        payload = self._payload(response)
        if self.long_form and response.status_code == 202:
            job_id = payload.get("id")
            if not isinstance(job_id, str) or not job_id:
                raise PermanentError("OtterAI did not return a long-form job ID")
            # Poll the configured API URL so a response cannot redirect the API key.
            return self._poll(job_id, headers)
        return self._text(payload)
