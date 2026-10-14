import os
import requests
from typing import Optional

from openai import OpenAI

from . import APIProvider, register

# gpt-4o-transcribe treats `language` as a soft hint and often answers Hindi
# speech in Urdu (Arabic) script, so pin the script with a default prompt.
DEFAULT_PROMPTS = {
    "hi": "हिंदी में देवनागरी लिपि में लिखें। Transcribe verbatim in Hindi, using Devanagari script.",
}


@register("openai")
class OpenAIProvider(APIProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _create(self, model_variant, audio_file, language, prompt):
        kwargs = dict(
            model=model_variant,
            file=audio_file,
            response_format="text",
            language=language,
            temperature=0.0,
        )
        if prompt is None:
            prompt = DEFAULT_PROMPTS.get(language)
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self.client.audio.transcriptions.create(**kwargs)

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
            response = requests.get(sample["row"]["audio"][0]["src"])
            # SDK infers the audio format from the filename, so pass a (name, bytes) tuple
            transcription = self._create(
                model_variant, ("audio.wav", response.content), language, prompt
            )
        else:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self._create(model_variant, audio_file, language, prompt)
        return transcription.strip()
