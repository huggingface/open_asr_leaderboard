import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register


@register("p99lab")
class P99LabProvider(APIProvider):
    """p99lab Audrey. Model variants: "audrey-0.7". Env: P99LAB_API_KEY, P99LAB_API_URL."""

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        base = os.environ["P99LAB_API_URL"].rstrip("/")
        headers = {"Authorization": f"Bearer {os.environ['P99LAB_API_KEY']}"}
        data = {"model": model_variant, "language": language or "en"}
        if use_url:
            audio_bytes = requests.get(sample["row"]["audio"][0]["src"], timeout=60).content
            files = {"file": ("audio", audio_bytes)}
        else:
            with open(audio_file_path, "rb") as f:
                files = {"file": (os.path.basename(audio_file_path), f.read())}
        r = requests.post(f"{base}/v1/transcribe", headers=headers, files=files, data=data, timeout=300)
        if r.status_code in (400, 401, 413):
            raise PermanentError(f"p99lab API returned {r.status_code}: {r.text}")
        r.raise_for_status()
        return r.json()["text"]
