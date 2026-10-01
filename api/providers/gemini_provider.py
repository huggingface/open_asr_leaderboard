import base64
from typing import Optional

import requests
from google import genai

from . import APIProvider, register


# Dataset language code -> BCP-47 locale from the Gemini transcribe docs.
# Spanish has no es-ES entry in the supported-locales table, so es-419 is used.
LANGUAGE_CODES = {
    "en": "en-US",
    "de": "de-DE",
    "fr": "fr-FR",
    "it": "it-IT",
    "es": "es-419",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "hi": "hi-IN",
    "hy": "hy-AM",
}


@register("gemini")
class GeminiProvider(APIProvider):
    def __init__(self):
        # Reads GEMINI_API_KEY from the environment.
        self.client = genai.Client()

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if use_url:
            audio_bytes = requests.get(sample["row"]["audio"][0]["src"]).content
        else:
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

        # Audio is sent inline rather than via client.files.upload, which would
        # add an upload (and cleanup) request per sample.
        response = self.client.interactions.create(
            model=model_variant,
            input=[
                {
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "mime_type": "audio/wav",
                }
            ],
            generation_config={
                "transcription_config": {
                    "language_codes": [LANGUAGE_CODES.get(language, language)]
                }
            },
        )
        return response.output_text.strip()
