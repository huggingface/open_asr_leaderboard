# Speech Expert: Armenian ASR API profiles

This draft proposes exactly two Armenian profiles. The streaming entry is the
pure epoch-7 FP32 Zipformer, with its own previews and final text. Large is a
separate file-processing pipeline. Both use explicit model/language selection;
neither receives benchmark references, dataset names, prompts or vocabulary hints.
The updated pure Zipformer deployment has passed bounded public-route checks.
Large's empty-tail fix is deployed, and the previously failing full file passed
a separate public recheck. Full benchmark evaluation remains outstanding.

| Model ID | Processing | Entry-level price |
|---|---|---|
| `speech-expert/SpeechExpert-hy-stream` | Pure Armenian Zipformer through native WebSocket | 16.80 RUB/hour (~$0.2045/hour) |
| `speech-expert/SpeechExpert-STT-HY-Large` | File recognition with local and external components | 600 RUB/hour (~$7.3047/hour) |

The adapter supports Armenian (`hy`) only. Pure Zipformer costs 7 credits/minute
and Large 250 credits/minute at the entry-level 0.04 RUB/credit. Streaming rounds
to a started second; Large has a 15-second minimum per request. USD figures use
82.1386 RUB/USD effective 2026-11-07, queried 2026-11-09; billing is in RUB,
not a fixed USD tariff. See [API/pricing](https://speech-expert.ru/documentation/api/stt-v1/)
and the [dated exchange rate](https://www.cbr.ru/scripts/XML_daily.asp?date_req=09.11.2026).

## Configuration and usage

Set `SPEECH_EXPERT_API_KEY` in the environment. Optional
`SPEECH_EXPERT_BASE_URL` defaults to `https://speech-expert.ru/api`.
Credentials are sent in headers, not URLs. Dependencies (`requests`,
`websockets`, `soundfile`, NumPy and SciPy via librosa) already exist in the
API evaluator image. Start with one worker.

The current multilingual launcher includes Armenian FLEURS and MCV under
`Metric-AI/open-asr-leaderboard-multilingual-datasets`. From the repository root
in the API evaluator environment:

```bash
PYTHONPATH=api:. python api/run_eval_ml.py \
  --dataset_path=Metric-AI/open-asr-leaderboard-multilingual-datasets \
  --config_name=fleurs_hy --language=hy --split=test \
  --model_name=speech-expert/SpeechExpert-hy-stream \
  --max_workers=1 --max_samples=25
```

Use `speech-expert/SpeechExpert-STT-HY-Large` separately with the same settings.
Calls incur API usage. A small sample is not a full leaderboard score. No
English entry, other-language entry, or new-dataset proposal is included.

## Pure streaming protocol and decoder

The streaming entry connects to `/api/stt/v1/ws` and sends:

```json
{"language":"hy-AM","model":"Zipformer-Streaming","privacy_mode":true,"external_ai_consent":false,"enable_automatic_punctuation":false,"diarization":false}
```

The Armenian resolver preserves this explicit generic API model. Its native
stream branch uses no final recognizer: Zipformer supplies both previews and
finals. It does not invoke Mhery, Large, an external recognizer, or optional LLM
punctuation. This differs from the service's separate `Mhery-RNNT` route.

The verified deployment identity is `hy-zipformer-epoch7-fp32-k2-p04`:
Icefall Zipformer v1, epoch-7 FP32, CPU ONNX Runtime. Greedy decoding permits
two symbols per encoder frame and subtracts 0.4 from blank logit 0 on every
joiner query. Initial predictor history is `[-1,0]`; blank 0 and UNK 2 end
the current frame. Each 39-frame input window advances 32 feature frames,
emits eight encoder frames and carries 35 recurrent cache tensors. The
80-bin streaming Kaldi frontend uses 25 ms/10 ms Povey windows, dither 0,
snip-edges false, low frequency 20 Hz and high frequency -400 Hz. No AGC is
enabled. Exact graph and source-checkpoint hashes are in the inventory.

Production endpointing resets decoder state at completed phrases: 2.4 seconds
of initial blank output, 1.2 seconds of trailing blanks after tokens, or a
24-second source-audio limit. EOF/hard cuts receive a server-side 0.66-second
zero tail for decoder context; that padding is not source audio or billing.
This endpointed operating mode differs from the archive's no-reset reference.
Core token/frame parity was checked with endpoints disabled, separately from
service endpoint/reset validation.

The client decodes WAV, mixes to mono and resamples once with a fixed polyphase
filter when necessary. It sends PCM16 little-endian at 16 kHz in 100 ms frames
paced in real time, reading events during upload. The client adds no silence.
After `{"event":"eof"}`, it waits for server close code 1000. Only final
texts enter the score; an empty final discards its preceding partial. Repeated
final identities replace one phrase; identical text in distinct intervals is
retained. Errors, malformed events, unfinished previews and unexpected closes
fail the request, rather than becoming partial successful predictions.

The scorer receives one final transcript and total elapsed time; **real-time
pacing is included in that time**. Upstream `--streaming` concerns dataset
loading, not real-time ASR. Optional `last_stream_metrics` use first PCM send
as the origin: first nonempty partial, first/last final, EOF and clean close.
`partial_events` includes empty previews. `last_final_after_eof_seconds` is a
signed offset and can be negative, not an unconditional positive latency.
See [WebSocket documentation](https://speech-expert.ru/documentation/api/websocket/).

## Large file protocol

Large calls `/api/stt/v1` with `model=SpeechExpert-STT-HY-Large`, `lang=hy`,
privacy mode and explicit external-processing consent. Optional LLM correction,
entities, diarization, emotions, deepfake analysis and summaries are disabled.
These options do not remove the model's intrinsic audio reconciliation.
The filename is fixed to `audio.wav`; raw `result` is returned unchanged.
An unexpected returned model identity is rejected.

The file timeout is 10 seconds connect / 600 seconds read. Streaming has a
bounded final-drain deadline and closes the socket on cancellation. Upstream
currently retries transient failures up to ten times after the first attempt;
the adapter adds no second retry loop. Retries may repeat paid requests. Error
messages omit response bodies, credentials and transport details. Inspect
failed/empty predictions before interpreting WER.

## Validation: separate diagnostics, not leaderboard results

The updated local CPU backend matched the archive's token IDs and encoder-frame
IDs on five fixed public Armenian FLEURS clips, using the same features without
endpoint resets. Streaming Kaldi-native-fbank and the archive's torchaudio
frontend also produced identical decoded token/frame sequences on those clips.

A separate in-process NativeSession test retained production endpoint resets.
All four quiet clips (-52 to -57 dBFS RMS) now produced previews without gain
adjustment and ended as single phrases, rather than the old deployment's empty
previews and short cuts. Silence and very low-level noise produced no text;
fragmented PCM preserved text and source cursors. These are local functional
checks, not public-network timings, independent accuracy evidence or full WER.
The fixed five-clip set totals 66.84 seconds and was used previously in local
development. The selected decoding profile itself has historical FLEURS/CV
development evidence, so no unseen-data or zero-shot claim is made.

After deployment, all five pure streaming requests completed successfully
through the public API. The worker identity matched the deployed manifest;
its configuration fingerprint remained unchanged throughout the check:
`dc488c1c533520930fb2613d13ad4d4effa7236289b864bfa061f630fa12c632`.
Public final text matched local results after trimming outer whitespace;
one clip retained a trailing space from an empty final phrase, and raw output
was preserved for scoring. Every clip produced a nonempty preview.

| Client-observed public diagnostic | Median | Range across five clips |
|---|---:|---:|
| First nonempty partial after **first PCM send** | 1.093 s | 0.813–2.000 s |
| Last final event after **explicit EOF send** | 141 ms | 109–172 ms |

These origins exclude connection setup and are not end-of-speech or first
audible-response latency. The last final may contain empty text; this metric
tracks the terminal final event. Audio was sent with real-time 100 ms pacing.
This small, previously used set is not a full leaderboard evaluation.

Historical Large validation completed 4/5 clips and returned HTTP 502 on the
fifth after 33.074 seconds. The post-rollout check again completed 4/5, with
HTTP 502 after 29.719 seconds on the same clip. Both failed attempts are retained,
without retry or replacement by a successful result. Diagnosis isolated a quiet
0.85-second trailing window: all three ASR hypotheses were empty, and the audio
reconciliation response stopped normally but omitted its message content.
The parser rejected that shape. The earlier diagnostic did not distinguish
missing content from explicit null; a later full-file probe confirmed omission.
A narrow all-empty-ASR handling fix passed a separate full-file in-process
check with live ASR/reconciliation backends in 26.649 seconds; the empty tail
was accepted and source hashes remained unchanged. This was not a request to
the deployed public API. After the API restart, one separate public request
for the full previously failing file succeeded in **112.765 seconds**, returning
280 characters. There was no adapter retry. A separate operational ledger
recorded two successful reconciliation calls of 92.813 and 8.750 seconds,
each on its first attempt. The remaining 11.202 seconds cannot be separated
into ASR and other overhead from the available logs. API source hashes and the worker fingerprint
remained unchanged during the request. This verifies that case after the fix,
not a fresh five-clip successful rerun; the original 4/5 reports are retained.

## Attribution and release gates

The pure model uses [Icefall's Zipformer architecture](https://github.com/k2-fsa/icefall).
The bundled Apache-2.0 LICENSE covers architecture code; it does not by itself
establish a redistribution license for weights or training data. This API
integration does not distribute weights. Training/data provenance and release
rights require explicit documentation before any separate model publication.

Large combines local Zipformer/Mhery with external Scribe hypotheses and
external audio reconciliation. Its Mhery component is
[mheryerznka/stt_arm_fastconformer_hybrid_large_no_pc](https://huggingface.co/mheryerznka/stt_arm_fastconformer_hybrid_large_no_pc),
Apache-2.0, revision `222f645143a6433eb81946ed0d4d30eb4d46501d`.
That model card lists FLEURS training data without a split; possible evaluation
overlap remains unresolved. Large is not represented as one independently
trained Speech Expert model. Final submission must freeze every component.

**No guarantee of no logging, storage, retention, inspection or training is
asserted.** Privacy mode does not prove all workers/external processors meet
the maintainers' private-audio requirement. Do not send private evaluation audio
until confirmed separately for each profile. Dedicated maintainer credentials,
full public WER/RTFx runs and training-overlap disclosure remain outstanding.
See [model_inventory.json](model_inventory.json).

Existing interactive demo: https://huggingface.co/spaces/Strij911/speech-expert-mixed-language-asr
The demo alone does not establish the deployed model revision.

## Adapter tests

```bash
python -m unittest discover -s api/speech_expert/tests -p 'test_*.py' -v
```

Tests use the actual upstream provider interface/registry, omitting unrelated
vendor SDK imports. They make no network requests and check exact routing,
removed-profile rejection, raw text, final/partial handling, clean EOF, errors,
timeouts, cancellation, interval validation and redaction.
