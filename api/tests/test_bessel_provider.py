"""No model loading, network access, or paid inference in these contract tests."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


HERE = Path(__file__).parent
package = types.ModuleType("bessel_fixture_providers")
package.APIProvider = object
package.PermanentError = type("PermanentError", (Exception,), {})
package.register = lambda name: lambda cls: cls
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + ".bessel_provider", HERE.parent / "providers/bessel_provider.py")
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)


class Reply:
    def __init__(self, value, code=200):
        self.value, self.status_code = value, code

    def json(self):
        return self.value


class Session:
    def __init__(self, *, text="unaltered  Transcript!", lost_post=False, failure=None, bad_seal=False,
                 bad_result=False, rejected=False):
        self.headers = {}
        self.calls = []
        self.text, self.lost_post, self.failure = text, lost_post, failure
        self.bad_seal, self.bad_result, self.rejected = bad_seal, bad_result, rejected

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        assert kwargs["allow_redirects"] is False
        if url.endswith("evaluation-contract"):
            return Reply({"model": provider.ENGINE_MODEL,
                "profile_sha256": "wrong" if self.bad_seal else provider.PROFILE_SHA256,
                "inference_runtime_sha256": provider.RUNTIME_SHA256, "upload_audio_storage": "memory_only"})
        identity = self.headers["Idempotency-Key"]
        if method == "POST":
            if self.rejected:
                return Reply({}, 401)
            if self.lost_post:
                raise requests.ConnectionError("response lost")
            return Reply({"id": identity, "status": "queued"}, 202)
        if self.failure:
            return Reply({"id": identity, "status": self.failure})
        return Reply({"id": identity, "status": "complete", "result": {
            "model": provider.ENGINE_MODEL, "profile_sha256": provider.PROFILE_SHA256,
            "runtime_sha256": "wrong" if self.bad_result else provider.RUNTIME_SHA256, "text": self.text}})


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = Path(self.tmp.name) / "test.wav"
        self.audio.write_bytes(b"synthetic file: provider transports bytes without interpreting them")
        self.env = patch.dict(os.environ, {"BESSEL_API_URL": "https://bessel.test", "BESSEL_API_KEY": "k" * 40})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_call(self, session, **kwargs):
        with patch.object(provider.requests, "Session", return_value=session), patch.object(provider.time, "sleep"):
            return provider.BesselProvider().transcribe(provider.VARIANT, str(self.audio),
                {"text": "SECRET REFERENCE", "dataset": "must-not-be-sent"}, **kwargs)

    def test_raw_transcript_and_no_reference_forwarding(self):
        s = Session()
        self.assertEqual(self.run_call(s), s.text)
        posts = [c for c in s.calls if c[0] == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0][2]["data"], {"model": provider.ENGINE_MODEL, "language": "en"})
        self.assertNotIn("SECRET REFERENCE", repr(s.calls))
        self.assertEqual(posts[0][2]["files"]["file"][1], self.audio.read_bytes())

    def test_empty_failed_default_is_preserved(self):
        self.assertEqual(self.run_call(Session(text="")), "")

    def test_lost_post_response_recovers_original_job_without_second_post(self):
        s = Session(lost_post=True)
        self.assertEqual(self.run_call(s), s.text)
        self.assertEqual(sum(c[0] == "POST" for c in s.calls), 1)
        self.assertTrue(s.calls[-1][1].endswith(s.headers["Idempotency-Key"]))

    def test_terminal_failure_does_not_restart(self):
        s = Session(failure="interrupted")
        with self.assertRaises(package.PermanentError):
            self.run_call(s)
        self.assertEqual(sum(c[0] == "POST" for c in s.calls), 1)

    def test_wrong_seal_does_not_submit_audio(self):
        s = Session(bad_seal=True)
        with self.assertRaises(package.PermanentError):
            self.run_call(s)
        self.assertFalse(any(c[0] == "POST" for c in s.calls))

    def test_wrong_result_runtime_is_rejected(self):
        with self.assertRaises(package.PermanentError):
            self.run_call(Session(bad_result=True))

    def test_unexpected_client_failure_is_permanent_not_billable_retry(self):
        s = Session()
        with patch.object(s, "request", side_effect=RuntimeError("private detail")):
            with self.assertRaises(package.PermanentError) as caught:
                self.run_call(s)
        self.assertNotIn("private detail", str(caught.exception))

    def test_requests_are_not_cached_across_samples(self):
        first, second = Session(), Session()
        self.run_call(first)
        self.run_call(second)
        self.assertNotEqual(first.headers["Idempotency-Key"], second.headers["Idempotency-Key"])

    def test_error_does_not_expose_key_or_audio(self):
        with self.assertRaises(package.PermanentError) as caught:
            self.run_call(Session(rejected=True))
        self.assertNotIn("k" * 40, str(caught.exception))
        self.assertNotIn("synthetic file", str(caught.exception))

    def test_prompts_and_non_english_are_rejected(self):
        for kwargs in ({"prompt": "dataset hint"}, {"language": "fr"}, {"use_url": True}):
            with self.assertRaises(package.PermanentError):
                self.run_call(Session(), **kwargs)



if __name__ == "__main__":
    unittest.main()
