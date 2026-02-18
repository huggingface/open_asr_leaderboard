import os
import requests
from typing import Optional

from . import APIProvider, register


@register("aquavoice")
class AquaVoiceProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        api_key = os.getenv("AQUAVOICE_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError(
                "AQUAVOICE_API_KEY environment variable not set, go to https://withaqua.com/api-dashboard to create a key"
            )
        endpoint = "https://api.aquavoice.com/api/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                endpoint,
                files={"file": audio_file},
                data={"model": model_variant},
                headers=headers,
            )
        return response.json()["text"]
