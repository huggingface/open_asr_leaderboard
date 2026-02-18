import os
from typing import Optional

import assemblyai as aai

from . import APIProvider, register


@register("assembly")
class AssemblyAIProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        transcriber = aai.Transcriber()

        # Models like "universal-3-pro" use the newer speech_models (list) API
        MULTI_MODEL_VARIANTS = {"universal-3-pro"}
        if model_variant in MULTI_MODEL_VARIANTS:
            config = aai.TranscriptionConfig(
                speech_models=[model_variant],
                language_detection=True,
            )
        else:
            config = aai.TranscriptionConfig(
                speech_model=model_variant,
                language_code=language,
            )

        if use_url:
            audio_url = sample["row"]["audio"][0]["src"]
            audio_duration = sample["row"]["audio_length_s"]
            if audio_duration < 0.160:
                print(f"Skipping audio duration {audio_duration}s")
                return "."
            transcript = transcriber.transcribe(audio_url, config=config)
        else:
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )
            if audio_duration < 0.160:
                print(f"Skipping audio duration {audio_duration}s")
                return "."
            transcript = transcriber.transcribe(audio_file_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(
                f"AssemblyAI transcription error: {transcript.error}"
            )
        return transcript.text
