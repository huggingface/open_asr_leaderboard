import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from websockets.frames import Close

from provider_loader import load_provider

adapter, registry = load_provider()


class FakeSocket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent, self.timeouts = [], []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def send(self, item):
        self.sent.append(item)

    def recv(self, timeout=None):
        self.timeouts.append(timeout)
        value = next(self.events, ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True))
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value) if isinstance(value, dict) else value


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SPEECH_EXPERT_API_KEY": "private-test-key", "SPEECH_EXPERT_BASE_URL": "https://speech-expert.ru/api"})
        self.env.start()
        self.provider = adapter.SpeechExpertProvider()

    def tearDown(self):
        self.env.stop()

    def run_stream(self, events, pcm=b""):
        socket = FakeSocket(events)
        self.socket = socket
        with patch.object(adapter, "_read_pcm16", return_value=pcm), patch.object(adapter, "_open_socket", return_value=socket) as connect, patch.object(adapter.requests, "post") as file_request:
            result = self.provider.transcribe("SpeechExpert-hy-stream", "unused.wav", {"text": "reference-secret"}, language="hy")
            file_request.assert_not_called()
        self.assertEqual(connect.call_args.args, ("wss://speech-expert.ru/api/stt/v1/ws", "private-test-key"))
        return result

    def test_streaming_config_without_file_fallback(self):
        self.assertEqual(self.run_stream([{"text": "final", "is_final": True}]), "final")
        self.assertEqual(json.loads(self.socket.sent[0]), {"language": "hy-AM", "model": "Zipformer-Streaming", "privacy_mode": True, "external_ai_consent": False, "enable_automatic_punctuation": False, "diarization": False})
        self.assertEqual(json.loads(self.socket.sent[-1]), {"event": "eof"})
        self.assertTrue(self.socket.closed)
        self.assertNotIn("reference-secret", repr(self.socket.sent))
        self.assertTrue(all(0 < value <= 600 for value in self.socket.timeouts))
        self.assertEqual(self.provider.last_stream_metrics["pacing"], "real_time")
        self.assertEqual(self.provider.last_stream_metrics["final_events"], 1)

    def test_partial_replaced_by_final_and_duplicate_id_replaced(self):
        result = self.run_stream([
            {"text": "draft", "is_final": False, "phrase_id": 0},
            {"text": "next draft", "is_final": False, "phrase_id": 1},
            {"text": "one", "is_final": True, "phrase_id": 0},
            {"text": "one corrected", "is_final": True, "phrase_id": 0},
            {"text": "two", "is_final": True, "phrase_id": 1},
        ])
        self.assertEqual(result, "one corrected two")
        self.assertEqual(self.provider.last_stream_metrics["partial_events"], 2)

    def test_same_text_in_different_ranges_is_not_deduplicated(self):
        self.assertEqual(self.run_stream([
            {"text": "yes", "is_final": True, "start_ms": 0, "end_ms": 100},
            {"text": "yes", "is_final": True, "start_ms": 0, "end_ms": 100},
            {"text": "yes", "is_final": True, "start_ms": 100, "end_ms": 200},
        ]), "yes yes")

    def test_empty_final_discards_nonempty_partial(self):
        self.assertEqual(self.run_stream([{"text": "hallucination", "is_final": False}, {"text": "", "is_final": True}]), "")

    def test_empty_partials_do_not_count_as_first_readable_text(self):
        self.assertEqual(self.run_stream([{"text": " ", "is_final": False}, {"text": "", "is_final": False}, {"text": "", "is_final": True}]), "")
        self.assertNotIn("first_partial_seconds", self.provider.last_stream_metrics)
        self.assertEqual(self.provider.last_stream_metrics["partial_events"], 2)

    def test_invalid_phrase_interval_fails(self):
        for bounds in ({"start_ms": -1, "end_ms": 3}, {"start_ms": 4, "end_ms": 3}, {"start_ms": 0}, {"start_ms": "0", "end_ms": 3}):
            with self.subTest(bounds=bounds), self.assertRaises(RuntimeError):
                self.run_stream([{"text": "x", "is_final": True, **bounds}])

    def test_partial_only_completion_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_stream([{"text": "draft", "is_final": False}])
        self.assertTrue(self.socket.closed)

    def test_json_error_fails_even_after_successful_final(self):
        for event, exception in (({"error": "secret", "status": 400}, registry.PermanentError), ({"error": "secret"}, RuntimeError)):
            with self.subTest(event=event), self.assertRaises(exception) as raised:
                self.run_stream([{"text": "partial success", "is_final": True}, event])
            self.assertNotIn("secret", str(raised.exception))

    def test_abnormal_close_and_early_clean_close_fail(self):
        with self.assertRaises(RuntimeError):
            self.run_stream([ConnectionClosedError(Close(1011, "private"), None, None)])
        with self.assertRaises(RuntimeError):
            self.run_stream([], pcm=b"\0" * 3200)
        self.assertTrue(self.socket.closed)

    def test_policy_close_is_permanent(self):
        with self.assertRaises(registry.PermanentError):
            self.run_stream([ConnectionClosedError(Close(1008, "private"), None, None)])

    def test_timeout_and_malformed_event_fail(self):
        for event in (TimeoutError("private"), "invalid", [], {"text": "x"}, {"text": "x", "is_final": "true"}):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                self.run_stream([event])

    def test_cancel_closes_socket_and_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_stream([KeyboardInterrupt()])
        self.assertTrue(self.socket.closed)

    def test_audio_resample_preserves_duration_and_pcm_format(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "8khz.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\0" * 800 * 4)
            result = adapter._read_pcm16(str(path))
        self.assertEqual(len(result), 1600 * 2)
        self.assertEqual(set(result), {0})


if __name__ == "__main__":
    unittest.main()
