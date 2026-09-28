import requests
from io import BytesIO
from typing import Optional

import openai

from . import APIProvider, register


@register("openai")
class OpenAIProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        if use_url:
            response = requests.get(sample["row"]["audio"][0]["src"])
            audio_data = BytesIO(response.content)
            response = openai.Audio.transcribe(
                model=model_variant,
                file=audio_data,
                response_format="text",
                language=language,
                temperature=0.0,
            )
        else:
            with open(audio_file_path, "rb") as audio_file:
                response = openai.Audio.transcribe(
                    model=model_variant,
                    file=audio_file,
                    response_format="text",
                    language=language,
                    temperature=0.0,
                )
        return response.strip()
