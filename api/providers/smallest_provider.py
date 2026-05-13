import os
import requests
from typing import Optional

from . import APIProvider, register


@register("smallestai")
class SmallestAIProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        api_key = os.getenv("SMALLESTAI_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError(
                "SMALLESTAI_API_KEY environment variable not set, get your key at https://console.smallest.ai"
            )
        endpoint = "https://api.smallest.ai/waves/v1/pulse/get_text"
        if use_url:
            audio_url = sample["row"]["audio"][0]["src"]
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": audio_url},
                params={"language": language},
            )
        else:
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/octet-stream",
                },
                data=audio_data,
                params={"language": language},
            )
        response.raise_for_status()
        return response.json().get("transcription", "")
