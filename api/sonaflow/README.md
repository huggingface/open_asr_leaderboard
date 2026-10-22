# Bessel v1.5

**English speech recognition. One API.**

Bessel v1.5 is a versioned English speech-to-text API from **Sonaflow Inc.**
Upload audio and receive its transcript through a single authenticated endpoint.
Use **`Sonaflow/bessel-v1.5`** as the model identifier in requests and evaluation results.

| Property | Value |
| --- | --- |
| Company | Sonaflow Inc. |
| Product | Bessel |
| Release | v1.5 |
| Model ID | `Sonaflow/bessel-v1.5` |
| Access | Hosted API; evaluation credentials provided privately |
| Language | English (`en`) |
| Input | WAV or FLAC; mono/stereo; 8–48 kHz; up to 300 seconds and 24 MiB |
| Output | Transcript text |
| Price ($/audio hour) | N/A — pricing not publicly announced |
| Documentation | https://besselaudio.com/model-card |

## API

Base URL: `https://api.besselaudio.com`. Authenticate with `Authorization: Bearer <API_KEY>`.
Keep credentials out of source control, screenshots and public logs.

```bash
export BESSEL_API_URL="https://api.besselaudio.com"
export BESSEL_API_KEY="<private-api-key>"
REQUEST_ID=$(python -c 'import uuid; print(uuid.uuid4().hex)')

curl "$BESSEL_API_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $BESSEL_API_KEY" \
  -H "Idempotency-Key: $REQUEST_ID" \
  -F "model=Sonaflow/bessel-v1.5" \
  -F "language=en" \
  -F "file=@audio.wav"
```

Example response (illustrative):

```json
{"model":"Sonaflow/bessel-v1.5","text":"Welcome to Bessel.","language":"en","duration":2.1}
```

Responses also include opaque `profile_sha256` and `runtime_sha256` version checksums.
The transcript is returned verbatim, including an empty string when applicable.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/models` | Model identity and input limits |
| GET | `/v1/evaluation-contract` | Version checksums and data-handling metadata |
| POST | `/v1/audio/transcriptions` | Upload and wait for the transcript |
| POST | `/v1/jobs` | Upload and receive a job ID (`202`) |
| GET | `/v1/jobs/{id}` | Recover or poll that job |

Jobs progress through `queued`, `running`, then `complete`, `failed` or `interrupted`.
A completed job contains the transcription under `result`. Use a new idempotency
key for each recording and evaluation run. If a response is lost, retrieve the
original job. Do not replay a terminal or interrupted request as a fresh job.

Only `file`, `model`, and `language` are supported. References, dataset identifiers,
vocabulary lists, and prompts are not accepted. One fixed configuration applies
across datasets. Authentication errors use `401`; invalid input uses `422`;
oversized uploads use `413`; conflicting idempotency keys use `409`;
a full queue uses `429`. Start evaluation with one worker.

## Open ASR Leaderboard integration

```bash
export HF_TOKEN="<write-token>"
export RESULTS_BUCKET="<your-results-bucket>"
MODEL="Sonaflow/bessel-v1.5 1" bash api/run_api.sh

# Offline adapter tests, with no transcription calls:
python -m unittest discover -s api/tests -p 'test_sonaflow_provider.py' -v
```

The adapter validates the service version before submitting audio, passes no
reference text or dataset hints, and preserves the raw returned transcript.
It recovers requests with their original job IDs and fails closed on a version
mismatch. The official runner and scorer remain unchanged.

## Evaluation evidence

The current ten public test splits contain **76,544 inputs**. Eight datasets use audited outputs collected on October 19–21, 2026. Both complete LibriSpeech splits use the public API, from 2026-10-22T08:50:43.490743+00:00 through 2026-10-22T11:27:17.159143+00:00. All results were scored with the current official scorer. This combines measurements from the stated dates; it is not a fresh full-suite run.
Scores are contributor-reported, using the unmodified official scorer at
`7e8d1a1003c201324e9003dd5bc34244340a63b0`. One fixed inference configuration is used throughout.

| Dataset | Inputs | WER (%) |
| --- | ---: | ---: |
| ami_cleaned | 7,805 | 8.91 |
| earnings22_cleaned_aa_chunked | 341 | 4.35 |
| gigaspeech_cleaned | 18,768 | 7.44 |
| monsoon_en_in | 2,102 | 2.97 |
| spgispeech | 39,341 | 2.85 |
| urgent2024 | 1,000 | 6.65 |
| urgent2024_clean | 1,000 | 3.73 |
| voxpopuli_cleaned_aa | 628 | 1.71 |
| librispeech:test.clean | 2,620 | 1.37 |
| librispeech:test.other | 2,939 | 2.73 |

The mean of the 10 official rounded dataset scores is **4.27100%**.
All 7 failed output records were retained. The official normalizer excluded
101 empty/ignored references (90 AMI and 11 GigaSpeech); these are reference filters.
Earnings22 uses official session-level regrouping.

All ten current public test splits are covered. The dataset revisions match the current official suite. Four historical timings include interruption recovery;
fresh end-to-end RTFx is not reported. No leaderboard rank or private-set score is
claimed. The configuration was frozen before the historical full run; public
results have since been inspected. Pretraining overlap is not established.
See [machine-readable results](bessel-v1.5-results.json).

## Evaluation data handling

The dedicated evaluation API processes uploaded audio in memory. Request IDs,
audio digests, job status and textual results are retained for durable recovery.
This is not a zero-retention claim for derived text or metadata. Processing may
involve contracted service providers; their applicable data-handling obligations
must also be covered by the evaluation arrangement.

Sonaflow Inc. commits not to train on, log, store, retain, or inspect Open ASR Leaderboard evaluation audio throughout the processing path. Audio is processed transiently in memory to fulfill the request. Private evaluation transcripts and derived labels will not be used for model development or tuning.
Use only the dedicated evaluation API for leaderboard recordings.

Commercial pricing has not been publicly announced. `N/A` does not mean free.
An API adapter and this documentation prepare an API submission; inclusion and
private-set verification are determined by the leaderboard maintainers.
