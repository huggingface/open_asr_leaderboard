import requests
import os
import base64
from typing import Optional

from . import APIProvider, PermanentError, register

MIME_MAP = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".webm": "audio/webm"}

# config.model per variant. None means the key is omitted entirely and the deployment
# serves whatever it defaults to -- which is what scribe_v1 is: the baseline, not a
# model named "scribe_v1". Sending an empty or wrong model is not equivalent.
VARIANT_MODELS = {
    "scribe_v1": None,
    "scribe_v2_pro": "zoom-scribe-en-pro",
}


@register("zoom")
class ZoomProvider(APIProvider):
    ENDPOINT = "https://api.zoom.us/v2/aiservices/scribe/transcribe"

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        api_key = os.getenv("ZOOM_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("ZOOM_API_KEY environment variable not set")

        # PermanentError, not a plain raise: transcribe_with_retry() would otherwise
        # spend 10 retries per sample on a name that cannot start working, and
        # run_eval.py records the exhausted sample as an empty hypothesis -- a whole
        # sweep of deletions with a plausible-looking WER instead of an error.
        if model_variant not in VARIANT_MODELS:
            raise PermanentError(
                f"Unknown zoom model 'zoom/{model_variant}'. Known: "
                + ", ".join(f"zoom/{name}" for name in VARIANT_MODELS)
            )

        if use_url:
            file_payload = sample["row"]["audio"][0]["src"]
            audio_duration = sample["row"]["audio_length_s"]
        else:
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()
            mime = MIME_MAP.get(os.path.splitext(audio_file_path)[1].lower(), "audio/wav")
            file_payload = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"

        if audio_duration <= 29.9:
            segmentation_mode = "none"
        else:
            segmentation_mode = "auto"

        config = {
            "language": "en-US",
            "segmentation_mode": segmentation_mode,
            "experimental_feature": {"gpu_pipeline_vnext": True}
        }
        model = VARIANT_MODELS[model_variant]
        if model is not None:
            config["model"] = model
        config["timestamps"] = True

        resp = requests.post(
            self.ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"file": file_payload, "config": config},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("result", {}).get("text_display", "") or "."
