# VoiceCodeBench

[VoiceCodeBench](https://huggingface.co/datasets/besimple-ai/voice-code-bench)
contains 300 English workplace recordings and 1,482 annotated entities. This
integration loads its pinned metadata and audio, runs NeMo Parakeet, and reports
Canonical Token/Entity Match (CTEM) separately from WER. Space display and published
leaderboard results still require maintainer integration. CTEM is a
**higher-is-better** percentage and must not enter the WER average; the generic
WER scorer rejects VCB manifests.

## Scoring protocol

CTEM = 100 × correctly recovered target entities / all target entities, pooled
across recordings. Empty transcripts remain in the denominator. Complete test
coverage is required unless `--allow-partial` explicitly marks a smoke test.
ASR receives audio only. Raw transcripts and stable audio IDs are preserved;
WER normalization never changes text before entity judging.

The judge is the downloadable, Apache-2.0
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint. Both model
and tokenizer are pinned to `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`.
The scorer starts its own loopback-only vLLM server and makes one call per
transcript, supplying the full text, lossless 256-character segments, and all
annotated targets. The rubric requires that a correct entity be clearly and
unambiguously derivable from information retained in the transcript. Gold
metadata cannot fill in missing content.

The [system prompt](normalizer/voice_code_bench_prompt.txt), strict target-identity
schema and citation validation are versioned in
[the implementation](normalizer/voice_code_bench_judge.py). The tested runtime is
`vllm/vllm-openai:v0.28.0-cu129`, BF16, a 32,768-token context, and the `qwen3`
reasoning parser. Requests use temperature zero, thinking enabled with `xhigh`
reasoning effort, an 8,192-token thinking budget, and a 16,384-token completion
limit including reasoning. At most 16 requests run concurrently by default.
Malformed, truncated, wrong-model or failed responses stop scoring. Evidence is
reconstructed from validated inclusive segment ranges. There is one judge path,
with no hosted inference API, API key, response repair or automatic inference
retry. Downloading weights and running GPU compute still have resource costs.

Annotations, audio and the entity-counting metric remain pinned to dataset
revision `bef2824f83ef1c796f3e79731a3b0741708730df`. This release has no annotated
alternative spellings; the adapter explicitly supplies an empty list per target.
Historical GPT scores must not be relabeled as scores from this protocol.
Re-evaluate saved transcripts and retain raw response caches. Cache headers bind
model/runtime identity, implementation and prompt; incompatible caches fail.

## Completed judge evaluation

The actual pinned BF16 weights ran on one NVIDIA H200 using the runtime above.
HF Jobs could not provision because the testing account lacked prepaid credits,
so inference ran in a private Modal GPU sandbox. The HF launcher itself has been
checked with mocked submissions; an end-to-end HF GPU job remains unverified.

On a frozen benchmark of 693 manually assessed entity decisions, the selected
single-call configuration scored **629/693**, compared with the saved GPT judge's
**597/693**. It made 43 incorrect accepts and 21 incorrect rejects and passed
**70/70 synthetic controls**. The completed development workload contained 616
saved transcript cases, 3,367 real entity decisions, and 181 recordings. All 656
unique responses passed strict validation; 133 exact successful pilot responses
were reused and 523 calls were new, with no retries.

All disagreements and 100 fixed agreement audits were covered by prior frozen
labels or masked full-transcript review. Manual comparison favored Qwen on 77
changes and GPT on 49, net +28. The recording-cluster 95% interval was
[−0.158, +1.783] percentage points and includes parity. The 100 agreement audits
found no shared errors; 35 other previously labeled, model-weighted agreements
were wrong for both. Agreement with GPT is not accuracy.

These are reused development data and Codex transcript-only assessments, not
independent human/audio ground truth. An optional full 7,200-transcript expansion
was stopped after the completed development evidence was accepted as sufficient
to proceed. No full-corpus accuracy, full-corpus parity or matched cost advantage
is claimed. Median new-call latency was 68.8 seconds at concurrency 16; the
58.6-minute worker run included 11.2 minutes of startup and checkpoint uploads.

## HF Jobs

Install the submission client locally and set `HF_TOKEN` or run `hf auth login`:

```bash
pip install 'huggingface_hub>=1.27.0,<2'
RESULTS_BUCKET="<your-results-bucket>" bash nemo_asr/submit_jobs_voice_code_bench.sh
```

The launcher runs `nvidia/parakeet-tdt-0.6b-v3` with batch size 128 on an H200,
waits for `COMPLETED`, then submits grading as a separate H200 job with the tested
vLLM image. Failed inference never starts grading. The local launcher must stay
running to coordinate the handoff. Both jobs receive only the HF credential for
Hub access and the results bucket. Job URLs, IDs and status are recorded.

Each run has a unique directory:

```text
hf://buckets/<bucket>/runs/<run-id>/<model-folder>/<manifest>.jsonl
```

Grading downloads the pinned weights and metric helpers and writes
`<manifest>.ctem.json` and `<manifest>.qwen27b-verifier-cache.json` beside the
manifest. Each validated raw response, including model identity and usage, is
checkpointed and uploaded. Explicitly resuming grading reuses completed responses;
failed requests do not become negative judgments. Run only one grading job per
manifest at a time. Replay uses a lightweight CPU job, starts no model server,
and fails on any missing response.

After successful grading, artifacts download to
`nemo_asr/results/voice_code_bench/<run-id>/<model-folder>/`. Set `DOWNLOAD_RESULTS=0`
to leave them in the bucket. `ORG_NAME` selects the job namespace; `FLAVOR`
selects inference hardware. Grading defaults are `h200`,
`vllm/vllm-openai:v0.28.0-cu129`, and an eight-hour timeout. `SCORING_FLAVOR`,
`SCORING_IMAGE`, and `SCORING_TIMEOUT` can change deployment resources; preserve
the pinned BF16 model and vLLM configuration when comparing scores.

For a two-recording smoke test:

```bash
MAX_EVAL_SAMPLES=2 RESULTS_BUCKET="<your-results-bucket>" \
  bash nemo_asr/submit_jobs_voice_code_bench.sh
```

To grade a saved manifest without rerunning ASR, pass its relative bucket path:

```bash
RESULTS_BUCKET="<your-results-bucket>" \
  bash nemo_asr/submit_jobs_voice_code_bench_scoring.sh \
  'runs/<run-id>/<model-folder>/<manifest>.jsonl'
```

Set `ALLOW_PARTIAL=1` for a smoke-test manifest. Default `VERIFIER_MODE=live-fill`
starts the local GPU model only if responses are missing. Set
`VERIFIER_MODE=replay` to submit a CPU replay job using a complete existing cache.
Fresh inference runs require `live-fill`.

## Local execution

Use the repository's NeMo environment for audio inference. Install the scoring
dependencies and the pinned metric package:

```bash
pip install -r requirements/requirements_jobs.txt
pip install -r requirements/requirements_voice_code_bench_scoring.txt
hf download besimple-ai/voice-code-bench --repo-type dataset \
  --revision bef2824f83ef1c796f3e79731a3b0741708730df \
  --include pyproject.toml --include README.md --include LICENSE \
  --include 'scripts/voice_code_bench/**' --local-dir .cache/voice-code-bench
pip install --no-deps .cache/voice-code-bench
PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}" python nemo_asr/run_eval.py \
  --model_id nvidia/parakeet-tdt-0.6b-v3 \
  --dataset_path besimple-ai/voice-code-bench --dataset default --split test \
  --device 0 --batch_size 128
```

Inference writes a raw manifest under `results/`. Run live grading in the tested
vLLM GPU image with the repository and results mounted, the scoring dependencies
and pinned metric package installed, and an H200 available. The scorer owns port
8000 on loopback; do not start another server there. It downloads the pinned
weights itself. For example, inside that environment:

```bash
python3 -m normalizer.voice_code_bench \
  --manifest results/MODEL_nvidia-parakeet-tdt-0.6b-v3_DATASET_besimple-ai-voice-code-bench_default_test.jsonl \
  --verifier-cache results/voice_code_bench.qwen27b-verifier-cache.json \
  --verifier-mode live-fill
```

Add `--allow-partial` for smoke tests. `--verifier-mode replay` (the CLI default)
needs neither vLLM nor a GPU when all responses are cached. `--verifier-workers`
sets request concurrency, default 16. Keep VCB manifests out of generic WER result
directories. Reports record dataset and model revisions, runtime configuration,
configuration digest, coverage, CTEM counts and per-entity decisions. Retain the
manifest, report and cache for auditing. Judge latency is excluded from ASR RTFx.

## Historical published baseline

The dataset's [published Parakeet predictions](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/baselines/predictions/modal_nvidia_parakeet_tdt_0_6b_v3.json)
score 1,165/1,482 = 78.60999% CTEM under the historical GPT verifier. These are
existing Modal/Transformers results, not scores from this judge or NeMo hardware
measurements. The dataset is test-only. Non-audio materials use MIT; audio also
prohibits voice cloning. See the dataset's
[license](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/LICENSE).
