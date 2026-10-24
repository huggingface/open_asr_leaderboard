import json
import os
import time
import wave
from typing import Optional

import requests

from . import APIProvider, PermanentError, register

DEFAULT_ENDPOINT = "https://api.meta.ai/v1/asr/transcribe"
DEFAULT_REALTIME_ENDPOINT = "wss://api.meta.ai/v1/asr/realtime"

# The leaderboard model id carries no version; the API wants the full one.
# The "-streaming" variant is the same model over the realtime WebSocket API.
MODEL_VARIANT_TO_API_MODEL = {
    "muse-voice-transcribe": "muse-voice-transcribe-1.0",
    "muse-voice-transcribe-streaming": "muse-voice-transcribe-1.0",
}
STREAMING_VARIANTS = {"muse-voice-transcribe-streaming"}

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

# Documented limit for file transcription (realtime caps the session at 60 min).
MAX_AUDIO_SECONDS = 10 * 60

# Realtime: sample rate -> the audioEncoding the handshake must declare.
SAMPLE_RATE_TO_ENCODING = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}
FRAME_MS = 80

# VERY IMPORTANT: the default mode `PUSH_TO_TALK` leads to much worse results
# docs: https://dev.meta.ai/docs/speech-to-text#modes
DEFAULT_STREAM_MODE = "ENDPOINTING"
MAX_TRAILING_SILENCE_SECONDS = 10.0


def _env(name: str, default: str) -> str:
    """os.getenv, but an empty value counts as unset.

    run_api*.sh forwards these as -e VAR="${VAR:-}", so an unset variable arrives
    as an empty string rather than absent. Plain os.getenv(name, default) would
    then return "" and, for `mode`, send "mode": "" -- which the API rejects with
    HTTP 400 "Malformed request".
    """
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@register("meta")
class MetaProvider(APIProvider):
    """Meta Muse Voice speech-to-text.

    Docs: https://dev.meta.ai/docs/speech-to-text

    Two transports for the same model:

    * `meta/muse-voice-transcribe` -- file transcription. Multipart POST with a
      `request` JSON part and an `audio` part; the reply's top-level `transcript`
      holds the full text.
    * `meta/muse-voice-transcribe-streaming` -- realtime WebSocket. Handshake
      frame, paced PCM binary frames, trailing silence, then
      `{"type": "endStream"}`; the text arrives on `speechComplete`.
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

        api_key = os.getenv("META_API_KEY")
        if not api_key:
            raise PermanentError("META_API_KEY is not set.")

        if use_url:
            raise PermanentError(
                "Meta provider supports file mode only; run without --use_url."
            )

        if audio_file_path is None:
            raise PermanentError("audio_file_path is required in file mode.")

        api_model = MODEL_VARIANT_TO_API_MODEL[model_variant]
        language_name = LANGUAGE_CODE_TO_NAME.get((language or "").strip().lower())

        if model_variant in STREAMING_VARIANTS:
            return self._transcribe_realtime(
                api_model, audio_file_path, api_key, language_name
            )
        return self._transcribe_file(
            api_model, audio_file_path, sample, api_key, language_name
        )

    # ── File transcription (HTTP) ────────────────────────────────────────────
    def _transcribe_file(
        self, api_model, audio_file_path, sample, api_key, language_name
    ) -> str:
        endpoint = _env("META_ENDPOINT", DEFAULT_ENDPOINT)

        audio_duration = (
            len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        )
        if audio_duration > MAX_AUDIO_SECONDS:
            raise PermanentError(
                f"Clip is {audio_duration:.0f}s; the Meta file endpoint caps audio at "
                f"{MAX_AUDIO_SECONDS}s. Use a chunked dataset for long-form audio."
            )

        mode = _env("META_FILE_MODE", DEFAULT_STREAM_MODE).upper()
        config: dict[str, object] = {
            "mode": mode,
            "model": api_model,
            "audioEncoding": "WAV",
        }
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
        transcript = body.get("transcript") or ""
        if transcript:
            return transcript
        # In the segmenting modes the per-turn text lives in `turns`; the docs say
        # the top-level field is always the complete text, so this only catches
        # the case where it comes back empty.
        turns = body.get("turns") or []
        return " ".join(
            t.get("transcript", "") for t in turns if isinstance(t, dict) and t.get("transcript")
        )

    # ── Realtime transcription (WebSocket) ───────────────────────────────────
    def _transcribe_realtime(
        self, api_model, audio_file_path, api_key, language_name
    ) -> str:
        """Stream a clip over the realtime socket.
        """
        # Imported here so the file path keeps working when `websockets` is absent.
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise PermanentError(
                "The streaming variant needs the `websockets` package "
                "(pip install websockets)."
            ) from exc
        from websockets.exceptions import ConnectionClosed

        endpoint = _env("META_REALTIME_ENDPOINT", DEFAULT_REALTIME_ENDPOINT)
        mode = _env("META_STREAM_MODE", DEFAULT_STREAM_MODE).upper()
        # Real-time pacing is what the reference client does; set
        # META_STREAMING_PACE=0 to push frames as fast as the socket takes them
        # (much quicker, but no longer the configuration the numbers come from).
        pace = _env("META_STREAMING_PACE", "1") != "0"

        # run_eval.py writes mono 16-bit PCM WAVs, which is exactly the realtime
        # wire format, so the samples go out as-is with the header stripped.
        with wave.open(audio_file_path, "rb") as wav:
            channels, sample_width = wav.getnchannels(), wav.getsampwidth()
            sample_rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
        if channels != 1 or sample_width != 2:
            raise PermanentError(
                f"Realtime needs mono 16-bit PCM; got {channels}ch/{sample_width * 8}-bit."
            )
        if sample_rate not in SAMPLE_RATE_TO_ENCODING:
            raise PermanentError(
                f"Realtime accepts {sorted(SAMPLE_RATE_TO_ENCODING)} Hz; got {sample_rate}."
            )

        handshake: dict[str, object] = {
            "authorization": {"accessToken": f"Bearer {api_key}"},
            "audioEncoding": SAMPLE_RATE_TO_ENCODING[sample_rate],
            "model": api_model,
            "mode": mode,
            # Every partial carries the complete hypothesis, so partials replace
            # rather than append. DELTA would make each one a fragment.
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        # Sent so both transports get the same options and stay comparable;
        if language_name:
            handshake["languageBias"] = [language_name]

        frame_bytes = (sample_rate * FRAME_MS // 1000) * sample_width
        frame_seconds = FRAME_MS / 1000.0
        silence = b"\x00" * frame_bytes

        state = {"finals": [], "partial": "", "complete": False}

        def handle(raw):
            """Fold one server frame into `state`; True once the turn is done."""
            if isinstance(raw, (bytes, bytearray)):
                return False
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                return False

            kind = event.get("type")
            if kind == "error":
                raise RuntimeError(
                    f"Meta realtime error: {event.get('message')} "
                    f"(session {event.get('sessionId')})"
                )
            if kind == "speechComplete":
                # The post-processed text for the turn. This is the final answer
                # in ENDPOINTING/DIARIZATION.
                text = event.get("transcript") or ""
                if text:
                    state["finals"].append(text)
                state["partial"] = ""
                state["complete"] = True
            elif kind == "transcript":
                text = event.get("transcript") or ""
                if event.get("final") and mode not in ("ENDPOINTING", "DIARIZATION"):
                    # PUSH_TO_TALK completes here; in the other modes a final
                    # transcript is still a hypothesis superseded by
                    # speechComplete, so it is kept only as a fallback.
                    state["finals"].append(text)
                    state["partial"] = ""
                    state["complete"] = True
                else:
                    state["partial"] = text
            return state["complete"]

        def drain(socket, timeout):
            """Consume whatever is already queued without blocking the sender."""
            while True:
                try:
                    raw = socket.recv(timeout=timeout)
                except TimeoutError:
                    return False
                if handle(raw):
                    return True

        try:
            with connect(endpoint, open_timeout=30, close_timeout=30) as socket:
                socket.send(json.dumps(handshake))

                for offset in range(0, len(pcm), frame_bytes):
                    socket.send(pcm[offset : offset + frame_bytes])
                    if pace:
                        time.sleep(frame_seconds)
                    drain(socket, 0)

                # The endpointer needs to hear the speech stop. Without this the
                # turn never closes and speechComplete never arrives, which is
                # why cutting straight to endStream loses the post-processed text.
                deadline = time.monotonic() + MAX_TRAILING_SILENCE_SECONDS
                while not state["complete"] and time.monotonic() < deadline:
                    socket.send(silence)
                    if pace:
                        time.sleep(frame_seconds)
                    drain(socket, 0)

                # Closes the client-to-server stream; nothing may follow it.
                socket.send(json.dumps({"type": "endStream"}))

                # Keep reading until the server closes (1000 on success).
                while True:
                    try:
                        raw = socket.recv()
                    except ConnectionClosed:
                        break
                    handle(raw)
        except ConnectionClosed as exc:
            # A close before any result is transient; the retry loop re-sends.
            if not state["finals"] and not state["partial"]:
                raise RuntimeError(f"Meta realtime closed early: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Meta realtime connection failed: {exc}") from exc

        # Fall back to the last partial if the stream ended without a final one.
        return " ".join(state["finals"]) if state["finals"] else state["partial"]
