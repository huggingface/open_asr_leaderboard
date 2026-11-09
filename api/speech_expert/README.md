# Speech Expert: Armenian ASR API profiles

Two Armenian speech recognition profiles are available through this adapter:

| Model ID | Mode | Entry-level price |
|---|---|---|
| `speech-expert/SpeechExpert-hy-stream` | Streaming, with interim and final transcripts | 16.80 RUB/hour (~$0.2045/hour) |
| `speech-expert/SpeechExpert-STT-HY-Large` | File transcription | 600 RUB/hour (~$7.3047/hour) |

Supported language: Armenian (`hy`). Billing is in RUB. Streaming rounds up to
the next second; Large has a 15-second minimum per request. USD estimates use
82.1386 RUB/USD, effective 2026-11-07. See [pricing and API documentation](https://speech-expert.ru/documentation/api/stt-v1/)
and the [exchange rate](https://www.cbr.ru/scripts/XML_daily.asp?date_req=09.11.2026).

## Configuration and usage

Set `SPEECH_EXPERT_API_KEY` in the environment. Optional
`SPEECH_EXPERT_BASE_URL` defaults to `https://speech-expert.ru/api`.
Credentials are sent in headers. The required dependencies are included in
the API evaluator image.

Run from the repository root in the API evaluator environment:

```bash
PYTHONPATH=api:. python api/run_eval_ml.py \
  --dataset_path=Metric-AI/open-asr-leaderboard-multilingual-datasets \
  --config_name=fleurs_hy --language=hy --split=test \
  --model_name=speech-expert/SpeechExpert-hy-stream \
  --max_workers=1 --max_samples=25
```

For file transcription, replace the model with
`speech-expert/SpeechExpert-STT-HY-Large`. Start with one worker. Requests use
paid API credits; a short sample is not a full leaderboard evaluation.

## Adapter behavior

- The streaming profile sends audio in real time through the WebSocket API.
  **Reported elapsed time includes real-time pacing.** Only final transcripts
  are used for scoring; interim transcripts are available during recognition.
- Large uploads audio to the file transcription API. Optional enrichment,
  summarization and text correction are disabled.
- The adapter sends audio and fixed recognition settings. It does not send
  benchmark references, dataset names, prompts or vocabulary hints, and returns
  transcript text unchanged.
- Failed or incomplete requests raise errors. The adapter adds no retries;
  retries by the evaluation runner may incur additional charges.

See the [WebSocket API documentation](https://speech-expert.ru/documentation/api/websocket/)
and [profile inventory](model_inventory.json).

## Evaluation status

This is a draft submission. Adapter tests and public API smoke checks have
passed. Full Armenian WER/RTFx evaluation and dedicated maintainer credentials
remain pending.

Public FLEURS/Common Voice data has been used in development or training;
possible training/evaluation overlap remains under review. No unseen-data or
independent-training claim is made for these service profiles.

Large requires consent to external audio processing; the adapter supplies
that consent. Private-audio handling requirements have not yet been confirmed
for either profile. Do not send private evaluation audio until that review is
complete; privacy mode alone is not a retention or training guarantee.

## Tests and demo

```bash
python -m unittest discover -s api/speech_expert/tests -p 'test_*.py' -v
```

The tests make no network requests and cover routing, transcript handling,
stream completion and errors.

[Interactive demo](https://huggingface.co/spaces/Strij911/speech-expert-mixed-language-asr)
