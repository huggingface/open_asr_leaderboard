import io
import os
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from . import APIProvider, PermanentError, register

# Yoob ASR: OpenAI-compatible POST /v1/audio/transcriptions, English. The audio goes as the raw request body (16 kHz mono
# FLAC), options as query parameters. The API takes up to an hour per request, but the edge proxy closes requests that run
# past ~100 s, so recordings longer than CHUNK_S are sent in pieces cut at the quietest point within SEARCH_S seconds before
# each nominal boundary (the API itself cuts every request into <= 30 s pieces the same way).
# run_api.sh passes YOOB_API_URL even when unset (as ""), so an empty value means the default.
ENDPOINT = os.getenv("YOOB_API_URL") or "https://tts-api.yoob.com/v1/audio/transcriptions"
SR = 16000
CHUNK_S = 600
SEARCH_S = 30
FRAME_S = 0.1

VARIANT_MODELS = {"yoob-asr-2": "yoob-asr-2"}


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


@register("yoob")
class YoobProvider(APIProvider):
    def _post(self, model, audio, api_key):
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="FLAC")
        resp = requests.post(
            ENDPOINT,
            params={"model": model, "response_format": "text"},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "audio/flac"},
            data=buf.getvalue(),
            timeout=300,
        )
        if resp.status_code in (400, 401, 402, 403, 404):
            # Bad key, no credits, unknown model or undecodable audio: retrying cannot help.
            raise PermanentError(f"Yoob API {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()  # 429 / 5xx: transcribe_with_retry() retries
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
        api_key = os.getenv("YOOB_API_KEY")
        if not api_key:
            raise ValueError("YOOB_API_KEY environment variable not set")
        if model_variant not in VARIANT_MODELS:
            raise PermanentError(f"Unknown Yoob model 'yoob/{model_variant}'. Known: " + ", ".join(f"yoob/{m}" for m in VARIANT_MODELS))
        if language != "en":
            raise PermanentError("yoob-asr-2 is English only")

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

        model = VARIANT_MODELS[model_variant]
        bounds = _chunk_bounds(audio)
        texts = [self._post(model, audio[s:e], api_key) for s, e in zip(bounds[:-1], bounds[1:])]
        return " ".join(t for t in texts if t) or "."
