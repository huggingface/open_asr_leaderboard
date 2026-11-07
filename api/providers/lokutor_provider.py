import os
import requests
from typing import Optional

from . import APIProvider, PermanentError, register

# Lokutor Conv 2.0 is served by an ElevenLabs-Scribe-compatible endpoint:
# https://docs.lokutor.com (api-reference, speech-to-text). One model, nine languages
# (en, es, ca, gl, eu, fr, de, pt, it); the request names the language, which is the
# same for every dataset of a given language.
ENDPOINT = os.getenv("LOKUTOR_API_URL", "https://api.lokutor.com/v1/speech-to-text")

# ISO-639-1 (what the leaderboard passes) -> the ISO-639-3 codes the endpoint documents.
LANGUAGE_CODES = {
    "en": "eng", "es": "spa", "ca": "cat", "gl": "glg", "eu": "eus",
    "fr": "fra", "de": "deu", "pt": "por", "it": "ita",
}

VARIANTS = {"conv-2.0"}


@register("lokutor")
class LokutorProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        api_key = os.getenv("LOKUTOR_API_KEY")
        if not api_key:
            raise PermanentError("LOKUTOR_API_KEY environment variable not set")
        if model_variant not in VARIANTS:
            raise PermanentError(
                f"Unknown model 'lokutor/{model_variant}'. Known: "
                + ", ".join(f"lokutor/{v}" for v in sorted(VARIANTS))
            )
        lang3 = LANGUAGE_CODES.get(language)
        if lang3 is None:
            raise PermanentError(f"Language '{language}' is not served by lokutor/conv-2.0")

        if use_url:
            audio_bytes = requests.get(sample["row"]["audio"][0]["src"], timeout=120).content
        else:
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()

        # The endpoint decodes WAV; run_eval.py writes every sample as WAV.
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={
                "language_code": lang3,
                "timestamps_granularity": "none",
                "diarize": "false",
            },
            timeout=300,
        )
        if response.status_code in (401, 402, 403):
            raise PermanentError(f"Lokutor API refused the request ({response.status_code})")
        response.raise_for_status()
        return response.json()["text"]
