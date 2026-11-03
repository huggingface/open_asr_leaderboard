# VoiceCodeBench: CTEM integration

[VoiceCodeBench](https://huggingface.co/datasets/besimple-ai/voice-code-bench)
contains 300 English workplace recordings (5.587 hours), with 1,482 annotated
entities. It measures whether ASR preserves exact values such as phone numbers,
email addresses, file paths, and command flags.

This initial integration provides dataset loading, a NeMo Parakeet HF Jobs script,
and the official CTEM scorer. The Space display still needs maintainer
integration. CTEM is a separate **higher-is-better** percentage;
it must not enter the leaderboard's WER average. The generic WER scorer rejects
VoiceCodeBench manifests to prevent accidental inclusion.

## Scoring protocol

CTEM = 100 × correctly recovered target entities / all target entities.
Counts are pooled across recordings, rather than averaging recording scores.
An empty ASR transcript remains in the denominator. Full coverage is required
unless `--allow-partial` is explicitly set for a smoke test.

ASR receives audio only, with no target entities, domain hints, vocabulary,
or post-ASR correction. The NeMo harness saves raw output and stable audio IDs
through duration sorting. Neither references nor predictions pass through the
WER text normalizer in the VCB path: punctuation, casing, digits, and separators
can affect value recovery.

The scorer reuses the benchmark's official `openai_gpt_5_5_v1` verifier and
`score_entity_capture` function. Its prompt judges whether each exact canonical
value is recoverable from the transcript, accepting acoustic or written forms
when they express the same value. This is an LLM judgment, not string equality.
Code, annotations, and audio are pinned to dataset revision
`bef2824f83ef1c796f3e79731a3b0741708730df`. The hosted verifier model (`gpt-5.5`)
is mutable; pinning the prompt does not guarantee identical future judgments.
Keep the verifier cache to replay the exact recorded responses.

The versioned [verifier configuration](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/scripts/voice_code_bench/verifiers/openai_gpt_5_5_v1.json)
contains the complete prompt and response schema. The
[verifier implementation](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/scripts/voice_code_bench/verifier.py)
validates responses and keys cached judgments by transcript, targets, and
verifier configuration. A replay cache miss fails rather than calling an API.

## Run one model

Run the commands below from the repository root. Install the shared dependencies
and pinned VCB package locally for job submission and CTEM scoring:

```bash
pip install -r requirements/requirements_jobs.txt
hf download besimple-ai/voice-code-bench --repo-type dataset \
  --revision bef2824f83ef1c796f3e79731a3b0741708730df \
  --include pyproject.toml --include README.md --include LICENSE \
  --include 'scripts/voice_code_bench/**' \
  --local-dir .cache/voice-code-bench
pip install .cache/voice-code-bench
```

### HF Jobs

Set `HF_TOKEN` in your environment and choose a results bucket you can write to.
Set `OPENAI_API_KEY` locally to score automatically after inference; otherwise
the script downloads the manifests and prints commands for scoring later.

```bash
RESULTS_BUCKET="<your-results-bucket>" bash nemo_asr/submit_jobs_voice_code_bench.sh
```

The script runs `nvidia/parakeet-tdt-0.6b-v3` with batch size 16 on an H200 by
default. It injects the local evaluator and normalizer into the NeMo Space image,
downloads results to `nemo_asr/results/voice_code_bench/<model-folder>/`, and
scores CTEM locally. The verifier cache is stored at
`nemo_asr/results/voice_code_bench/voice_code_bench.verifier-cache.json`.
Use `ORG_NAME` for an organization job namespace or `FLAVOR` to select hardware.

To check just two recordings:

```bash
MAX_EVAL_SAMPLES=2 RESULTS_BUCKET="<your-results-bucket>" \
  bash nemo_asr/submit_jobs_voice_code_bench.sh
```

The script adds `--allow-partial` to scoring for this smoke test. To replay
existing verifier responses without an API key, set `VERIFIER_MODE=replay`;
replay requires a populated cache and fails on cache misses.

### Local inference

To run on local hardware instead, use the repository's NeMo environment (see
`requirements/requirements_nemo.txt`) and invoke the evaluator directly:

```bash
PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}" python nemo_asr/run_eval.py \
  --model_id nvidia/parakeet-tdt-0.6b-v3 \
  --dataset_path besimple-ai/voice-code-bench \
  --dataset default --split test \
  --device 0 --batch_size 1
```

Pass `--device -1` for CPU or adjust `--batch_size` for available memory. Add
`--max_eval_samples 2` for a smoke test. Record hardware and model revision when
publishing results. Audio is downloaded from Besimple's Hub dataset and resampled
to 16 kHz. Streaming dataset loading is not supported yet.

Local inference requires no verifier key. It writes a raw JSONL manifest under
`results/` and prints a scoring command. For a full local run, score with:

```bash
# Set OPENAI_API_KEY in your environment first. live-fill makes paid API
# requests for uncached transcripts, sending raw transcripts and target entities.
python -m normalizer.voice_code_bench \
  --manifest results/MODEL_nvidia-parakeet-tdt-0.6b-v3_DATASET_besimple-ai-voice-code-bench_default_test.jsonl \
  --verifier-cache results/voice_code_bench.verifier-cache.json \
  --verifier-mode live-fill
```

Add `--allow-partial` for the two-recording smoke test. Subsequent runs can use
`--verifier-mode replay` (the default), which needs no API key. Live-fill saves
each response as it finishes, allowing scoring to resume after an interruption.
Keep VCB manifests in a separate directory when scoring other datasets with
the generic WER scorer.

The `.ctem.json` report records CTEM, its numerator and denominator, coverage,
dataset revision, verifier configuration digest, and per-entity decisions.
The source manifest preserves transcripts and inference timing. Retain both
files and the verifier cache for auditing. Verifier latency is excluded from
the harness's RTFx.

## Published baseline

The benchmark publishes [Parakeet predictions and entity judgments](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/baselines/predictions/modal_nvidia_parakeet_tdt_0_6b_v3.json):
1,165 / 1,482 entities = 78.60999% CTEM. These are existing Modal/Transformers
results, not a new run of this NeMo harness or leaderboard hardware.

The dataset is test-only. Non-audio materials use MIT; audio additionally
prohibits voice cloning. See the dataset's
[license](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/LICENSE)
and [dataset card](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/DATASET_CARD.md).
