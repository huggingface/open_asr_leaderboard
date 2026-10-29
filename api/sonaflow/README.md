# Bessel Consensus3 r1

**English speech recognition. One API.**

Bessel Consensus3 r1 is a versioned English speech-to-text API from **Sonaflow Inc.**
Use **`Sonaflow/bessel-consensus3-r1`** as the model identifier in requests and results.

| Property | Value |
| --- | --- |
| Company | Sonaflow Inc. |
| Product | Bessel |
| Release | Consensus3 r1 |
| Model ID | `Sonaflow/bessel-consensus3-r1` |
| Access | Hosted API; evaluation credentials provided privately |
| Language | English (`en`) |
| Input | WAV or FLAC; mono/stereo; 8–48 kHz; up to 300 seconds and 24 MiB |
| Output | Normalized English transcript text; punctuation and capitalization are not preserved |
| Price ($/audio hour) | N/A — pricing not publicly announced |
| Documentation | [Bessel API](https://besselaudio.com/model-card) |

## API

Base URL: `https://api.besselaudio.com`. Authenticate with `Authorization: Bearer <API_KEY>`.
Keep credentials out of source control, screenshots and public logs.

```bash
export BESSEL_API_URL="https://api.besselaudio.com"
export BESSEL_API_KEY="<private-api-key>"
export BESSEL_MODEL="Sonaflow/bessel-consensus3-r1"
REQUEST_ID=$(python -c 'import uuid; print(uuid.uuid4().hex)')

curl "$BESSEL_API_URL/v1/jobs" \
  -H "Authorization: Bearer $BESSEL_API_KEY" \
  -H "Idempotency-Key: $REQUEST_ID" \
  -F "model=$BESSEL_MODEL" \
  -F "language=en" \
  -F "file=@audio.wav"

# Poll this same ID until complete, failed or interrupted:
curl "$BESSEL_API_URL/v1/jobs/$REQUEST_ID" \
  -H "Authorization: Bearer $BESSEL_API_KEY"
```

A completed job includes a `result` object with `model`, `text`, `language`,
`duration`, `profile_sha256` and `runtime_sha256`. For example, its text may be
`"welcome to bessel"`. The adapter preserves exactly what the API returns,
including an empty string. It does not restore punctuation or capitalization,
or apply additional normalization before saving the returned transcript.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/models` | Model identity and input limits |
| GET | `/v1/evaluation-contract` | Version checksums and data-handling metadata |
| POST | `/v1/audio/transcriptions` | Upload and wait for the transcript |
| POST | `/v1/jobs` | Upload and receive a job ID (`202`) |
| GET | `/v1/jobs/{id}` | Recover or poll that job |

Jobs progress through `queued`, `running`, then `complete`, `failed` or `interrupted`.
Use a new idempotency key for each recording and evaluation run. If a response
is lost, retrieve the original job. Do not replay a terminal or interrupted
request as a fresh job.

Only `file`, `model` and `language` are supported. References, dataset identifiers,
vocabulary lists and prompts are not accepted. One fixed English configuration
applies across datasets. Authentication errors use `401`; invalid input uses
`422`; oversized uploads use `413`; conflicting idempotency keys use `409`;
a full queue uses `429`. Start evaluation with one worker.

The service is configured to remain running during idle periods. Queueing and
processing still take time; deployments or instance recovery can require startup.
Prefer the asynchronous job interface and preserve the job ID after a timeout.

## Open ASR Leaderboard integration

```bash
export BESSEL_API_URL="https://api.besselaudio.com"
export BESSEL_API_KEY="<private-api-key>"
export HF_TOKEN="<write-token>"
export RESULTS_BUCKET="<your-results-bucket>"
MODEL="Sonaflow/bessel-consensus3-r1 1" bash api/run_api.sh

# Offline adapter tests, with no transcription calls:
python -m unittest discover -s api/tests -p 'test_sonaflow_provider.py' -v
```

The adapter validates the model and both version checksums before submitting
audio, passes no reference text or dataset hints, and preserves the returned
transcript. It recovers requests with their original job IDs and fails closed
on a version mismatch. The official data loaders, normalizer and scorer are unchanged.

| Version field | SHA-256 |
| --- | --- |
| Profile | `5c0bcaad39f97b37f6dd073e26dc65b41f6a3919af8befc6580cbe31b30d80c4` |
| Inference runtime | `84ad4385c75b5260bc8ba58eacb328112e2d34805a5bca30d224e88738d83899` |

## Evaluation evidence

These are **contributor-reported offline replay results**, produced on
October 22, 2026 (UTC) from archived public evaluation artifacts. They cover
**76,544 inputs across ten public test splits**. Scoring used the unmodified
official English normalizer and scorer at commit
`25e0b9e143b07eeaf3b50366c2525ddf0c0c9a6b`, with official reference filtering and
Earnings22 session regrouping. The sealed release reproduced the archived outputs
for all 76,544 inputs in a local replay check.

| Dataset | Inputs | WER (%) |
| --- | ---: | ---: |
| ami_cleaned | 7,805 | 8.13 |
| earnings22_cleaned_aa_chunked | 341 | 4.86 |
| gigaspeech_cleaned | 18,768 | 7.07 |
| monsoon_en_in | 2,102 | 2.62 |
| spgispeech | 39,341 | 2.34 |
| urgent2024 | 1,000 | 7.20 |
| urgent2024_clean | 1,000 | 3.74 |
| voxpopuli_cleaned_aa | 628 | 1.79 |
| librispeech:test.clean | 2,620 | 1.28 |
| librispeech:test.other | 2,939 | 2.49 |

The mean of the ten rounded per-dataset scores is **4.152%**. The mean before
per-dataset rounding is **4.153480531994278%**. These are equal-weight dataset
means, not a pooled word-weighted score.

The official reference filter excludes 101 empty/ignored references (90 AMI,
11 GigaSpeech), leaving 76,443 manifest rows. All 26 empty output strings in
those rows are retained for scoring. Empty output does not by itself establish
an API failure. Earnings22's 341 input chunks are regrouped into six sessions.

Public benchmark data and results were used during development and reviewed
before release selection. These scores are development evidence, not a held-out
assessment. Pretraining overlap is not established. No fresh full-suite audio
inference through this release's public endpoint has been performed; replay
parity does not establish fresh end-to-end API accuracy or speed. RTFx is not
reported. Maintainer private evaluation is pending, and no leaderboard rank or
independent verification is claimed.

See [machine-readable results](bessel-consensus3-r1-results.json) for aggregate
error counts and manifest checksums. Evaluation credentials are shared only
through an agreed private channel.

## Evaluation data handling

The dedicated evaluation API processes uploaded audio transiently in memory.
Request IDs, audio digests, job status and textual results are retained for
durable recovery; no fixed retention period is specified. This is not a
zero-retention claim for derived text or metadata.

Sonaflow Inc. commits not to train on, log, store, retain, or inspect Open ASR
Leaderboard evaluation audio throughout the processing path, including applicable
contracted processing services. Private evaluation transcripts and derived labels
will not be used for model development or tuning. Use only the dedicated
evaluation API for leaderboard recordings.

Commercial pricing has not been publicly announced. `N/A` does not mean free.
Leaderboard inclusion and private-set verification are determined by the maintainers.
