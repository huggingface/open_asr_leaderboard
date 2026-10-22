# Bessel Decision + CTC v5

**Submission status: draft integration, not an accepted leaderboard entry.**

Bessel v5 is an English transcription API by Sonaflow. It runs four managed Azure
Speech candidates (MAI Transcribe 2, MAI Transcribe 1.5, Speech Fast and LLM Speech)
plus pinned Whisper large-v3 and Qwen3-ASR-1.7B on Azure Container Apps T4 GPUs.
All inference executes in Azure. This endpoint is distinct from the Bessel website's
configurable comparison playground.

The system divides each recording into equal contiguous windows of at most 15
seconds, preserving silence and the final tail. A frozen wav2vec2 CTC model and
Microsoft Decision compare candidates. A challenger replaces MAI Transcribe 2 only
when it has a supported pairwise win against all five alternatives. Any failed
candidate retains the default; a failed default produces an empty hypothesis.
Selected chunk transcripts are joined without word-level fusion or extra rewriting.
The same configuration is used for every dataset. The API accepts no reference,
dataset name, prompt, vocabulary, or per-dataset selection setting.

## Identity and scope

- Provider alias: `bessel/decision-ctc-v5`.
- Frozen inference ID: `asrouter/decision-ctc-azure-serverless6-en-v5`.
- Profile SHA256: `a58b2ef869d532cde0c6c05bd4cf50843e60ffbad8050722295ec69009e04e10`.
- Historical inference-runtime SHA256: `65954fe37b49e0f5761e7d3616cfdc38e252f4d2daf25daaab78ae698c19b6d8`.
- The upload-only transport wrapper has a separate digest, exposed by
  `/v1/evaluation-contract`; it does not redefine the historical inference digest.
- Input: English WAV/FLAC, mono/stereo, 8–48 kHz, at most 300 seconds / 24 MiB.
- Output: the actual model transcript, including empty failed-default outputs.
- GPU services retain their existing 0–1 replica limits. Use one client worker for
  initial verification. Scaling and cold-start delays are part of API latency.
- Managed Azure Speech and Microsoft Decision versions are service-managed;
  their backend weights cannot be pinned. Open-source weight revisions are fixed.

## Historical public evaluation

The completed 2026-10-19/21 run used the eight-set API suite and unmodified scorer
at `huggingface/open_asr_leaderboard@25e0b9e143b07eeaf3b50366c2525ddf0c0c9a6b`.
It contains 70,985 input recordings, 71,480 internal chunks and 428,880 candidate
calls. All 199 failed candidates and seven failed selected outputs were retained.
The official normalizer excludes 101 empty/ignored references (90 AMI, 11 GigaSpeech);
these are disclosed reference filters, not removed failed hypotheses.

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

The mean of the eight official two-decimal dataset WER values is
**4.82625%**. This is
not the current ten-split leaderboard average. LibriSpeech `test.clean` and
`test.other` have not yet been evaluated for this submission. No leaderboard rank
or private-set result is claimed. Four historical requests include recovery after
interruption, so no fresh speed result is claimed from those timings.

No ASR or fusion model was trained in this submission. The v5 rules were frozen
before its full run; the team subsequently inspected public evaluation results.
This is not a claim of a new blind evaluation. Foundation-model pretraining overlap
cannot be ruled out. There is no reference lookup, sample cache shared across runs,
or use of the later v7/v28/v42 experimental selectors in this API.

## Evaluation API integration

Obtain the dedicated evaluation endpoint and credential privately. Never put a key
in a PR, script, committed environment file, model card, or public log.

```bash
export BESSEL_API_URL="https://<dedicated-evaluation-host>"
export BESSEL_API_KEY="<private-key>"
export HF_TOKEN="<write-token>"
export RESULTS_BUCKET="<your-results-bucket>"
MODEL="bessel/decision-ctc-v5 1" bash api/run_api.sh
```

Offline adapter contract tests (no model calls):

```bash
python -m unittest discover -s api/tests -p 'test_bessel_provider.py' -v
```

The provider checks the endpoint's frozen identity before sending audio. Each
invocation gets a fresh job ID. A lost response is recovered using the original ID;
terminal and ambiguous failures are not replayed as new paid inference. The
official runner records failures as empty hypotheses. Do not restart an interrupted
evaluation blindly: preserve its request IDs and results, recover via GET, and mark
interrupted timings. Do not substitute another model or drop failed rows.

## Data handling and access

The submission adds an upload transport boundary that checks the actual body size
and keeps multipart audio in memory, including files above Starlette's normal
spooling threshold. The original research endpoint must not be used as a substitute.
No new audio training, content logging, reference inspection, or persistent audio
archive is part of the submission. Durable recovery retains request IDs, digests,
statuses and textual results; this is disclosed separately from audio retention.

Deployment verification and upstream Azure Speech data-handling review are pending.
The leaderboard audio-handling guarantee must remain unchecked until these are
verified for the supplied endpoint. In particular, the Bessel website's 24-hour
audio-storage preview is **not** this evaluation path.

Entry-level customer price: **not yet published / pending owner confirmation**.
Evaluation access does not imply a free commercial tier. Actual Azure billing is
not fully reconciled and is not reported as zero. Website: https://www.besselaudio.com/.
