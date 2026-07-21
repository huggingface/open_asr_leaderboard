"""Public API provider for the Open ASR Leaderboard, targeting Spirelight's
live production TSP API. Submits audio via /v1/transcribe, polls
/v1/job/{job_id} for completion, and returns the transcript text.
"""
import os
import time
import requests

from . import register, APIProvider, PermanentError

API_BASE = os.environ.get("TSP_API_BASE", "http://transcribe.spirelight.net:29371")
API_KEY = os.environ.get("TSP_API_KEY", "")
POLL_INTERVAL_SEC = 2
POLL_TIMEOUT_SEC = 3600


@register("spirelight")
class TSPProvider(APIProvider):
    def transcribe(self, model_variant, audio_file_path, sample, use_url=False,
                   language="en", prompt=None):
        if use_url:
            raise PermanentError("use_url mode not supported")

        with open(audio_file_path, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/v1/transcribe",
                headers={"x-api-key": API_KEY},
                files={"audio": f},
                data={"language": language},
                timeout=120,
            )
        if resp.status_code != 200:
            raise PermanentError(f"Submit failed: {resp.status_code} {resp.text[:300]}")
        job_id = resp.json()["job_id"]

        start = time.time()
        while time.time() - start < POLL_TIMEOUT_SEC:
            status_resp = requests.get(
                f"{API_BASE}/v1/job/{job_id}",
                headers={"x-api-key": API_KEY},
                timeout=30,
            )
            if status_resp.status_code != 200:
                raise PermanentError(f"Status check failed: {status_resp.status_code} {status_resp.text[:300]}")
            data = status_resp.json()
            if data["status"] == "completed":
                return data.get("text", "")
            if data["status"] == "failed":
                raise PermanentError(f"Job failed: {data.get('error')}")
            time.sleep(POLL_INTERVAL_SEC)

        raise PermanentError(f"Timed out waiting for job {job_id}")
