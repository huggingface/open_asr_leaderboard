"""Armenian Speech Expert ASR adapter for huggingface/open_asr_leaderboard.

The variants are explicit service profiles, not a claim that all component
models were trained by Speech Expert. See api/speech_expert/model_inventory.json. Evaluation
audio privacy is a server/provider property; this client makes no retention
guarantee. In particular, do not use private evaluation audio until reviewed.
"""

import os
import json
import time
from typing import Optional
from urllib.parse import urlsplit

import requests

from . import APIProvider, PermanentError, register


DEFAULT_BASE_URL = "https://speech-expert.ru/api"
MODEL_LANGUAGES = {"SpeechExpert-hy-stream": "hy", "SpeechExpert-STT-HY-Large": "hy"}
EXTERNAL_MODELS = frozenset({"SpeechExpert-STT-HY-Large"})



@register("speech-expert")
class SpeechExpertProvider(APIProvider):
    """Explicit Armenian pure Zipformer streaming or Large file profile.

    Requires SPEECH_EXPERT_API_KEY. SPEECH_EXPERT_BASE_URL overrides the API
    root, for example https://speech-expert.ru/api. Transient errors are
    retried by the upstream runner, not by a second client retry loop.
    """

    def __init__(self):
        self.api_key = os.getenv("SPEECH_EXPERT_API_KEY", "").strip()
        if not self.api_key or self.api_key == "your_api_key":
            raise ValueError("SPEECH_EXPERT_API_KEY environment variable not set")
        self.base_url = (os.getenv("SPEECH_EXPERT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        parsed = urlsplit(self.base_url)
        local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            not parsed.hostname
            or not (parsed.scheme == "https" or local_http)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("SPEECH_EXPERT_BASE_URL must be an HTTPS API root (or local HTTP), without credentials/query/fragment")

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        self.last_stream_metrics = {}
        if model_variant not in MODEL_LANGUAGES:
            raise PermanentError("Unsupported Speech Expert model variant")
        expected_language = MODEL_LANGUAGES[model_variant]
        language = str(language or "").strip().lower().replace("_", "-").split("-", 1)[0]
        if language != expected_language:
            raise PermanentError("Language does not match the selected Speech Expert model variant")
        if use_url or not audio_file_path:
            raise PermanentError("Speech Expert requires local audio files; do not use --use_url")
        if prompt:
            raise PermanentError("This Speech Expert evaluation adapter does not accept prompts")
        if model_variant == "SpeechExpert-hy-stream":
            return self._transcribe_armenian_stream(audio_file_path)

        # Never read sample/reference text or pass a dataset-derived filename.
        # Only upstream-supplied WAV audio and fixed model/language settings are
        # sent. The raw transcript is returned unchanged for HF normalization.
        try:
            audio_file = open(audio_file_path, "rb")
        except OSError:
            raise PermanentError("Cannot open evaluation audio file") from None
        try:
            with audio_file:
                response = requests.post(
                    f"{self.base_url}/stt/v1",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"audio": ("audio.wav", audio_file, "audio/wav")},
                    data={
                        "model": model_variant,
                        "lang": language,
                        "privacyMode": "true",
                        "externalAiConsent": "true" if model_variant in EXTERNAL_MODELS else "false",
                        "llmCorrection": "false",
                        "entityExtraction": "false",
                        "diarization": "false",
                        "emotionAnalysis": "false",
                        "deepfakeAnalysis": "false",
                        "channelMode": "mono",
                        # No summarizationPreset: omission is the API's
                        # documented disabled setting, not a boolean field.
                    },
                    timeout=(10, 600),
                    allow_redirects=False,
                )
        except requests.RequestException as exc:
            # The upstream runner prints exception messages. Do not include
            # response bodies, request URLs, headers or third-party messages.
            raise RuntimeError(f"Speech Expert transport failure ({type(exc).__name__})") from None

        status = response.status_code
        if status in {408, 429} or 500 <= status <= 599:
            raise RuntimeError(f"Speech Expert temporary HTTP {status}")
        if not 200 <= status <= 299:
            raise PermanentError(f"Speech Expert HTTP {status}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("Speech Expert returned invalid JSON") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), str):
            raise RuntimeError("Speech Expert response has no string result field")
        if payload.get("model") is not None and payload["model"] != model_variant:
            raise PermanentError("Speech Expert response model differs from requested model")
        return payload["result"]

    def _transcribe_armenian_stream(self, audio_file_path: str) -> str:
        """Pure Armenian Zipformer previews/finals; no Mhery refinement."""
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        pcm = _read_pcm16(audio_file_path)
        endpoint = self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/stt/v1/ws"
        finals = {}
        pending = set()
        sequence = 0
        eof_sent = False
        sent_at = None
        metrics = {"pacing": "real_time", "chunk_ms": 100, "audio_duration_s": len(pcm) / 32000,
                   "partial_events": 0, "final_events": 0}
        self.last_stream_metrics = metrics

        def handle(raw):
            nonlocal sequence
            if not isinstance(raw, str):
                raise RuntimeError("Speech Expert returned a non-JSON streaming event")
            try:
                event = json.loads(raw)
            except ValueError:
                raise RuntimeError("Speech Expert returned invalid streaming JSON") from None
            if not isinstance(event, dict):
                raise RuntimeError("Speech Expert returned invalid streaming event")
            if "error" in event:
                if event.get("status") in {400, 401, 402, 403, 404, 422} or event.get("code") in {"insufficient_credits", "external_ai_consent_required"}:
                    raise PermanentError("Speech Expert rejected streaming configuration or credentials")
                raise RuntimeError("Speech Expert reported a streaming failure")
            if not isinstance(event.get("text"), str) or type(event.get("is_final")) is not bool:
                raise RuntimeError("Speech Expert streaming event has no text/finality")
            phrase_id = event.get("phrase_id")
            if phrase_id is not None and (type(phrase_id) is not int or phrase_id < 0):
                raise RuntimeError("Speech Expert returned an invalid phrase identity")
            if "start_ms" in event or "end_ms" in event:
                if (type(event.get("start_ms")) is not int or type(event.get("end_ms")) is not int
                        or not 0 <= event["start_ms"] <= event["end_ms"]):
                    raise RuntimeError("Speech Expert returned an invalid phrase interval")
            pending_key = ("phrase", phrase_id) if phrase_id is not None else ("current",)
            if not event["is_final"]:
                metrics["partial_events"] += 1
                if event["text"].strip():
                    metrics.setdefault("first_partial_seconds", time.monotonic() - sent_at)
                    pending.add(pending_key)
                else:
                    pending.discard(pending_key)
                return
            pending.discard(pending_key)
            metrics["final_events"] += 1
            now = time.monotonic() - sent_at
            metrics.setdefault("first_final_seconds", now)
            metrics["last_final_seconds"] = now
            if phrase_id is not None:
                key = ("phrase", phrase_id)
            elif type(event.get("start_ms")) is int and type(event.get("end_ms")) is int:
                key = ("range", event["start_ms"], event["end_ms"])
            else:
                key = ("sequence", sequence)
                sequence += 1
            # Repeated/refined finals for the same identity replace that phrase.
            # Identical text in different phrases must remain repeated.
            finals[key] = event["text"]

        try:
            with _open_socket(endpoint, self.api_key) as socket:
                socket.send(json.dumps({
                    "language": "hy-AM", "model": "Zipformer-Streaming",
                    "privacy_mode": True, "external_ai_consent": False,
                    "enable_automatic_punctuation": False, "diarization": False,
                }))
                sent_at = time.monotonic()
                deadline = sent_at + len(pcm) / 32000 + 600
                for offset in range(0, len(pcm), 3200):
                    socket.send(pcm[offset:offset + 3200])
                    target = sent_at + min(offset + 3200, len(pcm)) / 32000
                    # Read concurrently with paced audio transmission. The sync
                    # websocket client has its own I/O thread; no extra worker
                    # thread or orphan sender is created by this provider.
                    while time.monotonic() < target:
                        try:
                            handle(socket.recv(timeout=max(0, target - time.monotonic())))
                        except TimeoutError:
                            break
                socket.send(json.dumps({"event": "eof"}))
                eof_sent = True
                metrics["eof_sent_seconds"] = time.monotonic() - sent_at
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("Speech Expert streaming deadline exceeded")
                    handle(socket.recv(timeout=remaining))
        except ConnectionClosed as exc:
            code = getattr(getattr(exc, "rcvd", None), "code", None)
            if code != 1000 or not eof_sent:
                if code == 1008:
                    raise PermanentError("Speech Expert streaming policy rejection") from None
                raise RuntimeError("Speech Expert stream ended before clean completion") from None
            metrics["stream_closed_seconds"] = time.monotonic() - sent_at
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {400, 401, 402, 403, 404, 422}:
                raise PermanentError("Speech Expert WebSocket handshake rejected") from None
            raise RuntimeError("Speech Expert WebSocket handshake failed") from None
        except PermanentError:
            raise
        except Exception:
            # Upstream prints exception messages, so discard transport details.
            raise RuntimeError("Speech Expert streaming request failed") from None
        if pending:
            raise RuntimeError("Speech Expert stream ended with an unfinalized hypothesis")
        if "last_final_seconds" in metrics:
            metrics["last_final_after_eof_seconds"] = metrics["last_final_seconds"] - metrics["eof_sent_seconds"]
        return " ".join(finals.values())


def _open_socket(endpoint, api_key):
    from websockets.sync.client import connect
    return connect(endpoint, additional_headers={"Authorization": f"Api-Key {api_key}"},
                   open_timeout=10, close_timeout=10, ping_timeout=20,
                   max_size=1048576, max_queue=32)


def _read_pcm16(audio_file_path):
    """Decode WAV, mix to mono and resample once using a fixed polyphase filter."""
    import math
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    try:
        audio, rate = sf.read(audio_file_path, dtype="float64", always_2d=True)
        if rate <= 0 or not np.isfinite(audio).all():
            raise ValueError("Invalid audio")
        audio = audio.mean(axis=1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes()
    except Exception:
        raise PermanentError("Cannot decode evaluation audio") from None
