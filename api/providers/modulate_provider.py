import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register


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
        endpoint = os.getenv("MODULATE_ENDPOINT")
        if not endpoint:
            raise PermanentError(
                "MODULATE_ENDPOINT is not set. Export the full endpoint URL, e.g. "
                "https://platform.modulate.ai/api/velma-2-stt-batch-english-vfast"
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
        with open(audio_file_path, "rb") as fh:
            files = {"upload_file": (os.path.basename(audio_file_path), fh)}
            response = requests.post(
                endpoint, headers=headers, files=files, timeout=600
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
