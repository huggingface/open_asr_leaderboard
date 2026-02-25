import os
import requests
from io import BytesIO
from typing import Optional

from elevenlabs.client import ElevenLabs

from . import APIProvider, register


@register("elevenlabs")
class ElevenLabsProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        if use_url:
            response = requests.get(sample["row"]["audio"][0]["src"])
            audio_data = BytesIO(response.content)
            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id=model_variant,
                tag_audio_events=True,
            )
        else:
            with open(audio_file_path, "rb") as audio_file:
                transcription = client.speech_to_text.convert(
                    file=audio_file,
                    model_id=model_variant,
                    tag_audio_events=True,
                )
        return transcription.text
