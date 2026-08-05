import os
from typing import Optional

from gladiaio_sdk import GladiaClient

from . import APIProvider, register

# Default Gladia model when --model_name is "gladia" with no variant suffix.
DEFAULT_MODEL = "solaria-3"

# Leaderboard variant -> Gladia API model name.
# Add new entries here when a leaderboard id differs from the API model id.
MODEL_ALIASES: dict[str, str] = {
    "solaria-3": "solaria-3",
    # "solaria-4": "solaria-4",
}


def resolve_gladia_model(model_variant: str) -> str:
    """Map a leaderboard model variant to the Gladia API model id."""
    variant = (model_variant or DEFAULT_MODEL).strip()
    if variant in MODEL_ALIASES:
        return MODEL_ALIASES[variant]
    # Forward-compatible: accept new variants not yet listed above.
    return variant


@register("gladia")
class GladiaProvider(APIProvider):
    def __init__(self):
        api_key = os.environ.get("GLADIA_API_KEY")
        if not api_key:
            raise ValueError(
                "GLADIA_API_KEY is not set. Export it or add it to a .env file."
            )
        kwargs = {"api_key": api_key}
        region = os.environ.get("GLADIA_REGION")
        if region:
            kwargs["region"] = region
        self.client = GladiaClient(**kwargs)

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        if use_url:
            raise NotImplementedError("Gladia provider does not support --use_url")

        gladia_model = resolve_gladia_model(model_variant)
        response = self.client.prerecorded().transcribe(
            audio_file_path,
            {
                "model": gladia_model,
                "language_config": {"languages": [language]},
            },
        )
        if response.status != "done":
            raise RuntimeError(
                f"Gladia transcription failed with status={response.status!r}"
            )
        transcription = response.result and response.result.transcription
        if transcription is None or not transcription.full_transcript:
            return " "
        return transcription.full_transcript.strip()