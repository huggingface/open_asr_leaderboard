import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

# EESI Labs Inc. (https://eesi.ai): POST /v1/audio/transcriptions, multipart, OpenAI-compatible. With `store=false`
# the API keeps neither the audio nor its transcript. run_api.sh passes EESI_API_URL even when unset (as ""), so an
# empty value means the default.
ENDPOINT = (os.getenv("EESI_API_URL") or "https://api.eesi.ai").rstrip("/") + "/v1/audio/transcriptions"


@register("eesi-ai")
class EesiProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        api_key = os.getenv("EESI_API_KEY")
        if not api_key:
            raise ValueError("EESI_API_KEY environment variable not set")
        if use_url:
            src = sample["row"]["audio"][0]["src"]
            fetched = requests.get(src, timeout=120)
            fetched.raise_for_status()
            audio, name = fetched.content, "audio." + src.split("?")[0].rsplit(".", 1)[-1][:4]
        else:
            with open(audio_file_path, "rb") as f:
                audio, name = f.read(), os.path.basename(audio_file_path)
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": model_variant, "language": language or "en", "store": "false"},
            files={"file": (name, audio)},
            timeout=600,
        )
        if resp.status_code in (400, 401, 402, 403, 404, 413):
            # Bad key, no credits, unknown model, undecodable or oversized audio: retrying cannot help.
            raise PermanentError(f"EESI API {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()  # 429 / 5xx: transcribe_with_retry() retries
        return resp.json()["text"].strip()
