import io
import os
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from . import APIProvider, PermanentError, register

# Gabsy Ear: Gabsy's on-device speech model, served for evaluation at POST /v1/audio/transcriptions.
# The audio goes as the raw request body (16 kHz mono FLAC); the model name and response format are query
# parameters; the reply is plain text. The model reads up to 20 minutes per request; longer recordings are sent in
# pieces cut at the quietest point within SEARCH_S seconds before each nominal boundary.
# Transcription mode (the checklist's one per-dataset exception): verbatim by default; prompt "clean" selects the
# clean transcript mode (no repetitions or filler words, the convention of SPGISpeech's references), the same switch the
# Gabsy app offers as "Clean transcript". run_api.sh sets it for spgispeech only. Otherwise the model takes no prompt.
# run_api.sh passes GABSY_API_URL even when unset (as ""), so an empty value means the default.
ENDPOINT = os.getenv("GABSY_API_URL") or "https://gabsy.dev/v1/audio/transcriptions"
SR = 16000
CHUNK_S = 600
SEARCH_S = 30
FRAME_S = 0.1

VARIANT_MODELS = {"gabsy-ear": "gabsy-ear"}


def _chunk_bounds(audio):
    frame = int(FRAME_S * SR)
    n_frames = len(audio) // frame
    energy = np.square(audio[: n_frames * frame].reshape(n_frames, frame)).mean(axis=1)
    bounds, start = [0], 0
    while len(audio) - start > (CHUNK_S + SEARCH_S) * SR:
        target = (start + CHUNK_S * SR) // frame
        lo, hi = target - int(SEARCH_S / FRAME_S), min(target, n_frames - 1)
        cut = (lo + int(np.argmin(energy[lo : hi + 1]))) * frame
        bounds.append(cut)
        start = cut
    bounds.append(len(audio))
    return bounds


@register("gabsy")
class GabsyProvider(APIProvider):
    def _post(self, model, audio, api_key, mode="verbatim"):
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="FLAC")
        resp = requests.post(
            ENDPOINT,
            params={"model": model, "response_format": "text", "mode": mode},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "audio/flac"},
            data=buf.getvalue(),
            timeout=300,
        )
        if resp.status_code in (400, 401, 403, 404, 413):
            raise PermanentError(f"Gabsy API {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()  # 429 / 5xx are retried by transcribe_with_retry()
        return resp.text.strip()

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        api_key = os.getenv("GABSY_API_KEY")
        if not api_key:
            raise ValueError("GABSY_API_KEY environment variable not set")
        if model_variant not in VARIANT_MODELS:
            raise PermanentError(f"Unknown Gabsy model 'gabsy/{model_variant}'. Known: gabsy/gabsy-ear")
        if language not in (None, "", "en"):
            raise PermanentError("Gabsy Ear is English-only")
        if use_url:
            raw = requests.get(sample["row"]["audio"][0]["src"], timeout=120).content
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        elif audio_file_path is not None:
            audio, sr = sf.read(audio_file_path, dtype="float32")
        else:
            audio, sr = np.asarray(sample["audio"]["array"], dtype="float32"), sample["audio"]["sampling_rate"]
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        if prompt not in (None, "", "clean"):
            raise PermanentError("Gabsy Ear takes no prompt other than the transcription mode 'clean'")
        mode = "clean" if prompt == "clean" else "verbatim"
        model = VARIANT_MODELS[model_variant]
        bounds = _chunk_bounds(audio)
        texts = [self._post(model, audio[s:e], api_key, mode) for s, e in zip(bounds[:-1], bounds[1:])]
        return " ".join(t for t in texts if t) or "."
