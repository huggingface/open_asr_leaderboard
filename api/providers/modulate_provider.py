import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

MODEL_VARIANT_TO_ENDPOINT = {
    "vfast": "https://platform.modulate.ai/api/velma-2-stt-batch-english-vfast",
    "multilingual": "https://platform.modulate.ai/api/velma-2-stt-batch-multilingual-vfast",
}


@register("modulate")
class ModulateProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if model_variant not in MODEL_VARIANT_TO_ENDPOINT:
            raise PermanentError(
                f"Unknown Modulate model variant '{model_variant}'. "
                f"Known variants: {list(MODEL_VARIANT_TO_ENDPOINT)}"
            )

        endpoint = os.getenv(
            "MODULATE_ENDPOINT", MODEL_VARIANT_TO_ENDPOINT[model_variant]
        )
        api_key = os.getenv("MODULATE_API_KEY")
        if not api_key:
            raise PermanentError("MODULATE_API_KEY is not set.")

        # URL mode is not supported by the Modulate file-upload API.
        if use_url:
            raise PermanentError(
                "Modulate provider supports file mode only; run without --use_url."
            )

        audio_duration = (
            len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        )
        if audio_duration < 0.0:
            print(f"Skipping audio duration {audio_duration}s")
            return "."

        if audio_file_path is None:
            raise PermanentError("audio_file_path is required in file mode.")

        headers = {"X-API-Key": api_key}

        # Multilingual endpoint takes an optional `language` form field (short
        # ISO code, e.g. de/es/fr/it/pt); providing it selects the declared-
        # language path so the file routes straight to the language-best
        # transcriber. The English endpoint ignores the field.
        data = None
        if model_variant == "multilingual" and language:
            data = {"language": language.strip().lower()}

        with open(audio_file_path, "rb") as fh:
            files = {"upload_file": (os.path.basename(audio_file_path), fh)}
            response = requests.post(
                endpoint, headers=headers, files=files, data=data, timeout=600
            )

        # A 4xx is a request/auth problem - permanent, do not retry.
        if 400 <= response.status_code < 500:
            raise PermanentError(
                f"Modulate endpoint rejected the request "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            )

        # Any other non-200 (incl. 5xx / 502 from the ensemble gate) is transient;
        # raise a plain exception so run_eval.py's retry loop re-sends the clip.
        if response.status_code != 200:
            raise RuntimeError(
                f"Modulate endpoint HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        body = response.json()
        return body.get("text") or ""
