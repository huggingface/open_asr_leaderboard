"""HTTP adapter for the Open ASR leaderboard; contains no model implementation."""

import atexit
import threading
import ipaddress
import os
from pathlib import Path
import re
from typing import Optional
from urllib.parse import urlsplit
import uuid

import requests

from . import APIProvider, PermanentError, register

PREFIX = "U2"
LANGUAGES = frozenset({"de", "fr", "it", "es", "pt"})


_local = threading.local()
_sessions = []
_sessions_lock = threading.Lock()


def close_sessions():
    with _sessions_lock:
        for session in _sessions:
            session.close()
        _sessions.clear()
    if hasattr(_local, "session"):
        del _local.session


atexit.register(close_sessions)


def get_session():
    # The official runner creates provider instances per sample. Connections
    # therefore belong to worker threads, not individual provider instances.
    if not hasattr(_local, "session"):
        session = requests.Session()
        session.trust_env = False
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=1, max_retries=0, pool_block=True))
        _local.session = session
        with _sessions_lock:
            _sessions.append(session)
    return _local.session


class RetryableError(RuntimeError):
    def __init__(self, message, retry_after=1):
        super().__init__(message)
        self.retry_after = retry_after


def configuration():
    url = os.environ.get("U2_ASR_API_URL", "").strip()
    key = os.environ.get("U2_ASR_API_KEY", "").strip()
    version = os.environ.get("U2_ASR_MODEL_VERSION", "").strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise PermanentError("Set a valid HTTPS U2_ASR_API_URL") from None
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or "YOUR_" in url.upper() or parsed.hostname.endswith(".invalid")):
        raise PermanentError("Set a real HTTPS U2_ASR_API_URL without URL credentials")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise PermanentError("The official evaluator requires a publicly reachable API")
    if len(key) < 24 or any(s in key.upper() for s in ("YOUR_", "CHANGE_ME", "PLACEHOLDER")):
        raise PermanentError("Set U2_ASR_API_KEY using a private evaluation credential")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", version):
        raise PermanentError("Set an immutable U2_ASR_MODEL_VERSION")
    return url, key, version


@register(PREFIX)
class U2Provider(APIProvider):
    def transcribe(
        self, model_variant: str, audio_file_path: Optional[str], sample: dict,
        use_url: bool = False, language: str = "en", prompt: Optional[str] = None,
    ) -> str:
        # Never inspect/send sample: it can contain the ground-truth transcript.
        if use_url:
            raise PermanentError("This provider accepts local audio uploads only")
        if not isinstance(language, str):
            raise PermanentError("Language must be a two-letter string.")
        language = language.lower() if language.isascii() else language
        if language not in LANGUAGES:
            raise PermanentError("This API currently accepts only German (de), Italian (it), Portuguese (pt), Spanish (es), and French (fr). Use a two-letter language code; region tags are not accepted.")
        if prompt is not None:
            raise PermanentError("This fixed evaluation configuration does not accept prompts")
        url, key, version = configuration()
        if model_variant != version:
            raise PermanentError("The requested model does not match the frozen API version")
        path = Path(audio_file_path) if audio_file_path else None
        if path is None or not path.is_file():
            raise PermanentError("A local audio file is required")
        request_id = str(uuid.uuid4())
        # No nested retry loop: the official runner already owns retries.
        try:
            with path.open("rb") as stream:
                session = get_session()
                response = session.post(
                    url,
                    headers={"Authorization": "Bearer " + key, "X-Request-ID": request_id},
                    files={"file": ("audio.wav", stream, "audio/wav")},
                    data={"model": version, "language": language},
                    timeout=(60, 300), allow_redirects=False,
                )
        except requests.RequestException:
            raise RuntimeError("ASR API transport failure; sensitive details redacted") from None
        except OSError:
            raise PermanentError("The local audio file cannot be read") from None
        status = response.status_code
        if status != 200:
            # Do not print response bodies: they might expose provider internals.
            error = "ASR API returned HTTP " + str(status)
            if status in (408, 429) or 500 <= status < 600:
                try:
                    delay = max(1, min(60, int(response.headers.get("Retry-After", "1"))))
                except (ValueError, TypeError):
                    delay = 1
                raise RetryableError(error, retry_after=delay)
            raise PermanentError(error)
        try:
            body = response.json()
        except ValueError:
            raise PermanentError("ASR API returned invalid JSON") from None
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise PermanentError("ASR API response must contain a string text field")
        if body.get("model") != version or body.get("request_id") != request_id:
            raise PermanentError("ASR API version/request identity mismatch")
        return body["text"]  # A genuine empty transcript is valid; never normalize here.
