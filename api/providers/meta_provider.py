import json
import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

DEFAULT_ENDPOINT = "https://api.meta.ai/v1/asr/transcribe"

# The leaderboard model id carries no version; the API wants the full one.
MODEL_VARIANT_TO_API_MODEL = {
    "muse-voice-transcribe": "muse-voice-transcribe-1.0",
}

# `languageBias` takes language *names*, not ISO codes.
LANGUAGE_CODE_TO_NAME = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
}

# Documented limit for file transcription.
MAX_AUDIO_SECONDS = 10 * 60


@register("meta")
class MetaProvider(APIProvider):
    """Meta Muse Voice speech-to-text (file transcription).

    Docs: https://dev.meta.ai/docs/speech-to-text

    Multipart POST with two parts: `request` (a JSON config, sent with
    Content-Type application/json) and `audio` (the WAV bytes). The response is
    a single JSON document whose top-level `transcript` holds the full plain
    text -- speaker labels and timings only ever appear inside `turns`, so no
    stripping is needed here.

    Mode is pinned to PUSH_TO_TALK (the API default): single-turn plain
    transcription. ENDPOINTING and DIARIZATION only add a `turns` array, which
    this benchmark does not score, and would be a decoding change across runs.
    """

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if model_variant not in MODEL_VARIANT_TO_API_MODEL:
            raise PermanentError(
                f"Unknown Meta model variant '{model_variant}'. "
                f"Known variants: {list(MODEL_VARIANT_TO_API_MODEL)}"
            )

        endpoint = os.getenv("META_ENDPOINT", DEFAULT_ENDPOINT)
        api_key = os.getenv("META_API_KEY")
        if not api_key:
            raise PermanentError("META_API_KEY is not set.")

        # The file endpoint reads audio from the multipart body; there is no
        # URL mode (the realtime WebSocket API is a separate endpoint).
        if use_url:
            raise PermanentError(
                "Meta provider supports file mode only; run without --use_url."
            )

        if audio_file_path is None:
            raise PermanentError("audio_file_path is required in file mode.")

        audio_duration = (
            len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        )
        if audio_duration > MAX_AUDIO_SECONDS:
            raise PermanentError(
                f"Clip is {audio_duration:.0f}s; the Meta file endpoint caps audio at "
                f"{MAX_AUDIO_SECONDS}s. Use a chunked dataset for long-form audio."
            )

        config: dict[str, object] = {
            "mode": "PUSH_TO_TALK",
            "model": MODEL_VARIANT_TO_API_MODEL[model_variant],
            "audioEncoding": "WAV",
        }
        language_name = LANGUAGE_CODE_TO_NAME.get((language or "").strip().lower())
        if language_name:
            config["languageBias"] = [language_name]

        headers = {"Authorization": f"Bearer {api_key}"}

        with open(audio_file_path, "rb") as fh:
            files = {
                "request": (None, json.dumps(config), "application/json"),
                "audio": (os.path.basename(audio_file_path), fh, "audio/wav"),
            }
            response = requests.post(
                endpoint, headers=headers, files=files, timeout=600
            )

        # 429 is a documented rate limit, not a bad request: let run_eval.py's
        # retry loop back off and re-send rather than dropping the clip.
        if response.status_code == 429:
            raise RuntimeError(
                f"Meta endpoint rate limited (HTTP 429): {response.text[:200]}"
            )

        # Any other 4xx is a request/auth problem - permanent, do not retry.
        if 400 <= response.status_code < 500:
            raise PermanentError(
                f"Meta endpoint rejected the request "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            )

        # 5xx and anything else non-200 is transient.
        if response.status_code != 200:
            raise RuntimeError(
                f"Meta endpoint HTTP {response.status_code}: {response.text[:200]}"
            )

        body = response.json()
        return body.get("transcript") or ""
