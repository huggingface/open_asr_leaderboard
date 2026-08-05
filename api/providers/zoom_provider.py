import requests
import os
import base64
from typing import Optional

from . import APIProvider, register

MIME_MAP = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac"}


@register("zoom")
class ZoomProvider(APIProvider):
    ENDPOINT = "https://api.zoom.us/v2/aiservices/scribe/transcribe"

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        api_key = os.getenv("ZOOM_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("ZOOM_API_KEY environment variable not set")

        if use_url:
            file_payload = sample["row"]["audio"][0]["src"]
            audio_duration = sample["row"]["audio_length_s"]
        else:
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()
            mime = MIME_MAP.get(os.path.splitext(audio_file_path)[1].lower(), "audio/wav")
            file_payload = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"

        if audio_duration <= 29.9:
            segmentation_mode = "none"
        else:
            segmentation_mode = "auto"

        resp = requests.post(
            self.ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"file": file_payload,
                  "config": {
                    "language": "en-US",
                    "segmentation_mode": segmentation_mode,
                    "experimental_feature": {"model_pro": True},
                    "timestamps": True,
                  }},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("result", {}).get("text_display", "") or "."
