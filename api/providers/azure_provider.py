import requests
import os
import io
import json
from typing import Optional

from . import APIProvider, register

MIME_MAP = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac"}


@register("azure")
class AzureProvider(APIProvider):
    ENDPOINT = "https://eastus.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15"

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

        definition = {
            "enhancedMode": {
                "enabled": True,
                "task": "transcribe"
            }
        }
        if prompt is not None:
            # E.g., prompt = "Output must be in lexical format."
            definition["enhancedMode"]["prompt"] = [prompt]

        if use_url:
            file_url = sample["row"]["audio"][0]["src"]
            audio_resp = requests.get(file_url, timeout=120)
            audio_resp.raise_for_status()
            audio_data = io.BytesIO(audio_resp.content)
            files = {
                "audio": ("audio.wav", audio_data, "audio/wav"),
                "definition": (None, json.dumps(definition), "application/json"),
            }
        else:
            mime = MIME_MAP.get(os.path.splitext(audio_file_path)[1].lower(), "audio/wav")
            files = {
                "audio": (audio_file_path, open(audio_file_path, "rb"), mime),
                "definition": (None, json.dumps(definition), "application/json"),
            }

        resp = requests.post(
            self.ENDPOINT,
            headers={"Ocp-Apim-Subscription-Key": api_key},
            files=files,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("combinedPhrases", [{}])[0].get("text", "") or "."
