# VoiceCodeBench: CTEM integration

[VoiceCodeBench](https://huggingface.co/datasets/besimple-ai/voice-code-bench)
contains 300 English workplace recordings (5.587 hours), with 1,482 annotated
entities. It measures whether ASR preserves exact values such as phone numbers,
email addresses, file paths, and command flags.

This initial integration provides dataset loading, a NeMo Parakeet example,
and the official CTEM scorer. The Space display and HF Jobs configuration still
need maintainer integration. CTEM is a separate **higher-is-better** percentage;
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

Use the repository's NeMo environment (see `requirements/requirements_nemo.txt`),
then install the shared loading/scoring dependencies and pinned VCB package:

```bash
pip install -r requirements/requirements_jobs.txt
hf download besimple-ai/voice-code-bench --repo-type dataset \
  --revision bef2824f83ef1c796f3e79731a3b0741708730df \
  --include pyproject.toml --include README.md --include LICENSE \
  --include 'scripts/voice_code_bench/**' \
  --local-dir .cache/voice-code-bench
pip install .cache/voice-code-bench
bash nemo_asr/run_voice_code_bench.sh
```

The script runs `nvidia/parakeet-tdt-0.6b-v3` with batch size 1 on a GPU, using
the existing greedy decoding path. Pass `--device -1` for CPU or override
`--batch_size` for available memory. The conservative batch size is a starting
point, not an H200 throughput measurement. Record hardware and model revision
when publishing results. Audio is downloaded from Besimple's Hub dataset and
resampled to 16 kHz. Streaming dataset loading is not supported yet.

To check just two recordings:

```bash
bash nemo_asr/run_voice_code_bench.sh --max_eval_samples 2
```

Inference requires no verifier key. It writes a raw JSONL manifest under
`results/` and prints a scoring command. For a full run, score with:

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

## Validate without inference

```bash
pip install pytest
python -m pytest tests/test_voice_code_bench.py
```

The tests cover loading, raw text preservation, ID alignment, pooled scoring,
coverage validation, verifier replay, and exclusion from WER. The benchmark
also publishes [Parakeet predictions and entity judgments](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/baselines/predictions/modal_nvidia_parakeet_tdt_0_6b_v3.json):
1,165 / 1,482 entities = 78.60999% CTEM. These are existing Modal/Transformers
results, not a new run of this NeMo harness or leaderboard hardware.

The dataset is test-only. Non-audio materials use MIT; audio additionally
prohibits voice cloning. See the dataset's
[license](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/LICENSE)
and [dataset card](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/DATASET_CARD.md).
