import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import requests

from provider_loader import load_provider


adapter, registry = load_provider()


class SpeechExpertProviderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"SPEECH_EXPERT_API_KEY": "test-secret", "SPEECH_EXPERT_BASE_URL": "https://speech-expert.ru/api"})
        self.environment.start()
        self.directory = TemporaryDirectory()
        self.audio = Path(self.directory.name) / "reference-must-never-leak.wav"
        self.audio.write_bytes(b"RIFFtestfixture")
        self.provider = adapter.SpeechExpertProvider()

    def tearDown(self):
        self.directory.cleanup()
        self.environment.stop()

    def call(self, **kwargs):
        args = dict(model_variant="SpeechExpert-STT-HY-Large", audio_file_path=str(self.audio), sample={"text": "reference secret", "row": {"text": "reference secret"}}, language="hy")
        args.update(kwargs)
        return self.provider.transcribe(**args)

    def test_upstream_registry_and_interface(self):
        for expected in ("SpeechExpert-hy-stream", "SpeechExpert-STT-HY-Large"):
            with self.subTest(model=expected):
                provider, variant = registry.get_provider("speech-expert/" + expected)
                self.assertIsInstance(provider, registry.APIProvider)
                self.assertEqual(variant, expected)

    def test_exact_multipart_request_and_unnormalized_response(self):
        captured = {}
        def post(url, **kwargs):
            captured.update(url=url, **kwargs)
            captured["audio_bytes"] = kwargs["files"]["audio"][1].read()
            return Mock(status_code=200, json=Mock(return_value={"result": "  Twenty-one, €2.50!\n"}))
        with patch.object(adapter.requests, "post", side_effect=post):
            self.assertEqual(self.call(language="hy-AM"), "  Twenty-one, €2.50!\n")
        self.assertEqual(captured["url"], "https://speech-expert.ru/api/stt/v1")
        self.assertEqual(captured["headers"], {"Authorization": "Bearer test-secret"})
        self.assertEqual(captured["data"], {"model": "SpeechExpert-STT-HY-Large", "lang": "hy", "privacyMode": "true", "externalAiConsent": "true", "llmCorrection": "false", "entityExtraction": "false", "diarization": "false", "emotionAnalysis": "false", "deepfakeAnalysis": "false", "channelMode": "mono"})
        self.assertEqual(captured["audio_bytes"], b"RIFFtestfixture")
        self.assertEqual(captured["files"]["audio"][0], "audio.wav")
        self.assertTrue(captured["files"]["audio"][1].closed)
        self.assertFalse(captured["allow_redirects"])
        self.assertEqual(captured["timeout"], (10, 600))
        self.assertNotIn("reference secret", repr(captured))

    def test_each_variant_keeps_its_exact_model_and_language(self):
        for model, language in (("SpeechExpert-STT-HY-Large", "hy"),):
            with self.subTest(model=model), patch.object(adapter.requests, "post", return_value=Mock(status_code=200, json=Mock(return_value={"result": "ok"}))) as post:
                self.call(model_variant=model, language=language)
                self.assertEqual(post.call_args.kwargs["data"]["model"], model)
                self.assertEqual(post.call_args.kwargs["data"]["lang"], language)
                self.assertEqual(post.call_args.kwargs["data"]["externalAiConsent"], "true" if model in adapter.EXTERNAL_MODELS else "false")

    def test_response_model_mismatch_is_permanent(self):
        with patch.object(adapter.requests, "post", return_value=Mock(status_code=200, json=Mock(return_value={"result": "text", "model": "SpeechExpert-STT-UZ"}))):
            with self.assertRaises(registry.PermanentError):
                self.call()

    def test_exact_response_model_is_accepted(self):
        with patch.object(adapter.requests, "post", return_value=Mock(status_code=200, json=Mock(return_value={"result": "text", "model": "SpeechExpert-STT-HY-Large"}))):
            self.assertEqual(self.call(), "text")

    def test_invalid_variants_languages_url_and_prompts_fail_before_upload(self):
        for args in ({"model_variant": "Unsupported-ASR"}, {"language": "en"}, {"use_url": True}, {"audio_file_path": None}, {"prompt": "Read the reference"}):
            with self.subTest(args=args), patch.object(adapter.requests, "post") as post:
                with self.assertRaises(registry.PermanentError):
                    self.call(**args)
                post.assert_not_called()

    def test_english_profiles_are_explicitly_excluded(self):
        self.assertEqual(adapter.MODEL_LANGUAGES, {
            "SpeechExpert-hy-stream": "hy", "SpeechExpert-STT-HY-Large": "hy"})
        for model in ("SpeechExpert-STT-EN", "Unsupported-EN", "SpeechExpert-STT-RU", "SpeechExpert-STT-UZ-Large", "Unsupported-ASR"):
            with self.subTest(model=model), patch.object(adapter.requests, "post") as post:
                with self.assertRaises(registry.PermanentError):
                    self.call(model_variant=model, language="en")
                post.assert_not_called()

    def test_unsupported_streaming_profile_is_not_silently_remapped(self):
        with patch.object(adapter.requests, "post") as post, patch.object(adapter, "_open_socket") as socket:
            with self.assertRaises(registry.PermanentError):
                self.call(model_variant="Unsupported-streaming", language="hy")
            post.assert_not_called()
            socket.assert_not_called()

    def test_permanent_http_errors_do_not_leak_response(self):
        for status in (301, 302, 400, 401, 402, 403, 404, 413, 422):
            with self.subTest(status=status), patch.object(adapter.requests, "post", return_value=Mock(status_code=status, text="PRIVATE BODY test-secret")):
                with self.assertRaises(registry.PermanentError) as raised:
                    self.call()
                self.assertNotIn("PRIVATE", str(raised.exception))
                self.assertNotIn("test-secret", str(raised.exception))

    def test_transient_http_errors_are_left_to_upstream_retry(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status), patch.object(adapter.requests, "post", return_value=Mock(status_code=status)) as post:
                with self.assertRaises(RuntimeError):
                    self.call()
                self.assertEqual(post.call_count, 1)

    def test_network_errors_redacted_and_file_closed(self):
        for error in (requests.Timeout("PRIVATE test-secret"), requests.ConnectionError("PRIVATE test-secret")):
            captured = []
            def fail(*args, **kwargs):
                captured.append(kwargs["files"]["audio"][1])
                raise error
            with self.subTest(error=type(error)), patch.object(adapter.requests, "post", side_effect=fail):
                with self.assertRaises(RuntimeError) as raised:
                    self.call()
                self.assertNotIn("PRIVATE", str(raised.exception))
                self.assertNotIn("test-secret", str(raised.exception))
                self.assertTrue(captured[0].closed)

    def test_missing_file_is_permanent_without_path_disclosure(self):
        with self.assertRaises(registry.PermanentError) as raised:
            self.call(audio_file_path="private-path-does-not-exist.wav")
        self.assertNotIn("private-path", str(raised.exception))

    def test_malformed_success_is_not_silent_empty_prediction(self):
        for payload in ({}, {"result": None}, {"result": 42}, [], "text"):
            with self.subTest(payload=payload), patch.object(adapter.requests, "post", return_value=Mock(status_code=200, json=Mock(return_value=payload))):
                with self.assertRaises(RuntimeError):
                    self.call()

    def test_invalid_json_is_retryable(self):
        with patch.object(adapter.requests, "post", return_value=Mock(status_code=200, json=Mock(side_effect=ValueError("secret")))):
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                self.call()

    def test_legitimate_empty_transcript_is_preserved(self):
        with patch.object(adapter.requests, "post", return_value=Mock(status_code=200, json=Mock(return_value={"result": ""}))):
            self.assertEqual(self.call(), "")

    def test_missing_key_and_unsafe_base_urls_fail_early(self):
        with patch.dict(os.environ, {"SPEECH_EXPERT_API_KEY": ""}):
            with self.assertRaises(ValueError):
                adapter.SpeechExpertProvider()
        for url in ("http://remote.test/api/v1", "https://key@remote.test/api/v1", "https://remote.test?token=secret", "file:///tmp/test", "https://remote.test/#secret"):
            with self.subTest(url=url), patch.dict(os.environ, {"SPEECH_EXPERT_BASE_URL": url}):
                with self.assertRaises(ValueError):
                    adapter.SpeechExpertProvider()
        with patch.dict(os.environ, {"SPEECH_EXPERT_BASE_URL": "http://127.0.0.1:8001/api/v1/"}):
            self.assertEqual(adapter.SpeechExpertProvider().base_url, "http://127.0.0.1:8001/api/v1")


if __name__ == "__main__":
    unittest.main()
