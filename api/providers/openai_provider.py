import os
import tempfile
import requests
from typing import Optional

import numpy as np
import soundfile as sf
from openai import OpenAI

from . import APIProvider, register

# gpt-4o-transcribe treats `language` as a soft hint and often answers Hindi
# speech in Urdu (Arabic) script, so pin the script with a default prompt.
DEFAULT_PROMPTS = {
    "hi": "हिंदी में देवनागरी लिपि में लिखें। Transcribe verbatim in Hindi, using Devanagari script.",
}

# The API rejects uploads over 25 MB ("Audio file might be corrupted or unsupported"),
# so longer audio (e.g. the hour-long long-form recordings) is sent in chunks of about
# CHUNK_S seconds (~19 MB as 16 kHz 16-bit WAV), each cut at the quietest point within
# SEARCH_S seconds of its nominal boundary so that words are rarely split.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_S = 600
SEARCH_S = 30
FRAME_S = 0.1


def _chunk_bounds(audio, sr):
    """Sample offsets splitting `audio` into ~CHUNK_S pieces at low-energy points."""
    frame = int(FRAME_S * sr)
    n_frames = len(audio) // frame
    energy = np.square(audio[: n_frames * frame].reshape(n_frames, frame)).mean(axis=1)
    bounds, start = [0], 0
    # Cut only while more than CHUNK_S + SEARCH_S remains, so the last chunk is at
    # least SEARCH_S long (the API rejects very short audio) and still under the limit.
    while len(audio) - start > (CHUNK_S + SEARCH_S) * sr:
        target = (start + CHUNK_S * sr) // frame
        lo, hi = target - int(SEARCH_S / FRAME_S), min(target, n_frames - 1)
        cut = (lo + int(np.argmin(energy[lo : hi + 1]))) * frame
        bounds.append(cut)
        start = cut
    bounds.append(len(audio))
    return bounds


@register("openai")
class OpenAIProvider(APIProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _create(self, model_variant, audio_file, language, prompt):
        kwargs = dict(
            model=model_variant,
            file=audio_file,
            response_format="text",
            language=language,
            temperature=0.0,
        )
        if prompt is None:
            prompt = DEFAULT_PROMPTS.get(language)
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self.client.audio.transcriptions.create(**kwargs)

    def _create_chunked(self, model_variant, audio_file_path, language, prompt):
        audio, sr = sf.read(audio_file_path, dtype="float32")
        bounds = _chunk_bounds(audio, sr)
        texts = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            with tempfile.NamedTemporaryFile(suffix=".wav") as chunk:
                sf.write(chunk.name, audio[start:end], sr, format="WAV")
                with open(chunk.name, "rb") as chunk_file:
                    texts.append(self._create(model_variant, chunk_file, language, prompt).strip())
        return " ".join(t for t in texts if t)

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if use_url:
            response = requests.get(sample["row"]["audio"][0]["src"])
            # SDK infers the audio format from the filename, so pass a (name, bytes) tuple
            transcription = self._create(
                model_variant, ("audio.wav", response.content), language, prompt
            )
        elif os.path.getsize(audio_file_path) > MAX_UPLOAD_BYTES:
            transcription = self._create_chunked(model_variant, audio_file_path, language, prompt)
        else:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self._create(model_variant, audio_file, language, prompt)
        return transcription.strip()
