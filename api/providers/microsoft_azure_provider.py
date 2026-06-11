import requests
import os
import io
import json
from typing import Optional

from . import APIProvider, register

MIME_MAP = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac"}


@register("microsoft")
class MicrosoftAzureProvider(APIProvider):
    ENDPOINT = "https://northeurope.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15"

    # support 26 languages, list of locales in ML benchmark
    # It is Multi-lingual model, can use without specifying the language.
    LOCALE_DICT = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-PT",
    }

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None, 
    ) -> str:
        api_key = os.getenv("AZURE_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("AZURE_API_KEY environment variable not set")

        locale = self.LOCALE_DICT.get(language, "")
        definition = {
            "locales": [locale],
            "profanityFilterMode": "None",
            "enhancedMode": {
                "enabled": True,
                "task": "transcribe",
            },
        }
        if prompt is not None:
            # E.g., prompt = "Output must be in lexical format."
            definition["enhancedMode"]["prompt"] = [prompt]

        if use_url:
            file_url = sample["row"]["audio"][0]["src"]
            audio_resp = requests.get(file_url, timeout=120)
            audio_resp.raise_for_status()
            audio_data = io.BytesIO(audio_resp.content)
            files = [
                ("definition", (None, json.dumps(definition))),
                ("audio", ("audio.wav", audio_data, "audio/wav")),
            ]
        else:
            mime = MIME_MAP.get(os.path.splitext(audio_file_path)[1].lower(), "audio/wav")
            files = [
                ("definition", (None, json.dumps(definition))),
                ("audio", (audio_file_path, open(audio_file_path, "rb"), mime)),
            ]
        resp = requests.post(
            self.ENDPOINT,
            headers={"Ocp-Apim-Subscription-Key": api_key},
            files=files,
            timeout=300,
        )
        if not resp.ok:
            print(f"Azure API error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json().get("combinedPhrases", [{}])[0].get("text", "") or "."
