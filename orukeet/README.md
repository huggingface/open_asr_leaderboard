# Orukeet R15-0100: English short-form evaluation

This submission proposes [oruk/orukeet-r15-0100](https://huggingface.co/oruk/orukeet-r15-0100) as the earlier checkpoint for [PR #221](https://github.com/huggingface/open_asr_leaderboard/pull/221). R15-0100 is the archived parent of FT-4035. It predates the Monsoon-informed FT-4035 continuation and the subsequent r3 continuation trained directly on LibriSpeech test-other. The candidate was selected from its recorded lineage before the fresh evaluation.

**Earlier validation and selection exposure remains.** LibriSpeech test-other appeared in the initial adaptation/cooldown validation and evaluation, including all 2,939 recordings. Inspected initial logs show the monitored `val_wer` matching FLEURS, with test-other reported separately; whether test-other results influenced the fixed-step EMA export is unresolved. R15 recovery selection used a regression suite containing all 2,620 LibriSpeech test-clean recordings. Earlier VoxPopuli evaluation/selection also occurred; overlap with the current cleaned 628-record partition has not been established. Inherited pretraining overlap is incompletely verified. See [the full disclosure](training-disclosure.md).

We request maintainer confirmation that this rollback, with the remaining history disclosed, satisfies the requested checkpoint criterion. We do not describe it as fully unexposed or independently held out.

## Checkpoint

The model has a FastConformer encoder and TDT decoder, supports 25 languages, and uses CC BY-SA 4.0 weights. This submission evaluates English only. The exact checkpoint tensor audit verifies 627,008,134 model parameters, including 110,592 fixed Gabor coefficients. The count excludes 49,176 batch-normalization buffer scalars and 33,296 fixed preprocessing window/filterbank scalars from the 627,090,606 state-dictionary scalars.

| Identity | Value |
| --- | --- |
| Model | `oruk/orukeet-r15-0100` |
| Immutable revision | `073489c0619cd7939e327bebfc6c5d4ace4b69bf` |
| File | `orukeet-r15-0100.nemo` |
| SHA-256 | `4295a6d820a40b99786331d1c7a6b6c328916c8329b23d39415b0649a5d42811` |

## Reproduce

The launcher is pinned to the candidate Space revision and bundle below and verifies every bundled source file before invocation.

The [candidate evaluation Space](https://huggingface.co/spaces/oruk/open-asr-leaderboard-orukeet/tree/4c32209217703fdcbf474d54e68e62f3b3e5d804) contains the evaluator, unchanged pinned official normalizer, source patch, tests, model/data pins and bundle manifest. Its bundle SHA-256 is `590c29ff1945167a21dae22277e218064c0956ff71c5b7f8805716b3b6a96f69`.

Install `huggingface_hub>=1.11.0`, authenticate with a write token, and create a results bucket. From the repository root:

```sh
# Inspect the candidate's eight jobs without launching them.
bash orukeet/submit_jobs.sh --namespace YOUR_NAMESPACE --bucket YOUR_NAMESPACE/YOUR_BUCKET

# Execute the fixed plan with at most two H200 jobs concurrently.
bash orukeet/submit_jobs.sh --namespace YOUR_NAMESPACE --bucket YOUR_NAMESPACE/YOUR_BUCKET \
  --execute --receipt-directory ./orukeet-r15-job-receipts
```

The fixed profile uses the official NeMo Docker image by digest, H200, NeMo 2.7.2, Torch 2.8.0, BF16, batch 128, one loader worker, greedy-batch decoding and `max_symbols=10`. It preserves the official reference filtering and original audio boundaries. Each job has a 1,200-second limit. No automatic retries, checkpoint sweep or decoding sweep are included. Batch 128 follows the existing Parakeet profile; maximum-throughput tuning is not claimed.

CUDA synchronization brackets the full transcription pass, including file loading and decoding, after up to four warmup batches. Aggregate RTFx is total evaluated audio duration divided by total timed transcription duration across the eight datasets; it excludes model initialization, data staging and warmup. It is not the mean of per-dataset RTFx values or end-to-end job throughput.

Candidate results use the separate `oruk-orukeet-r15-0100/` bucket prefix. Retain all original manifests and completion receipts and require exactly one complete candidate run per dataset. Use the candidate collector and metadata validator from the pinned Space; they check model identity, dataset pins, output counts and artifact hashes before invoking the unchanged official scorer. Empty predictions remain in scoring. Earnings22's 341 ordered chunks are joined into six complete reference sessions.

## Fresh R15-0100 results

All eight complete runs passed artifact, count, input, runtime and independent numerical checks. Evaluated September 19, 2026 (UTC).

| Dataset | R15-0100 WER (%) | H200 RTFx |
| --- | ---: | ---: |
| AMI-Cleaned | 9.69 | 3625.50 |
| Earnings22-Cleaned-AA | 6.50 | 2980.66 |
| GigaSpeech-Cleaned | 7.95 | 5474.17 |
| LibriSpeech test-clean — earlier selection exposure | 1.49 | 4076.97 |
| LibriSpeech test-other — earlier validation/evaluation | 3.11 | 4144.22 |
| SPGISpeech | 3.39 | 6700.20 |
| Monsoon English | 3.98 | 4862.19 |
| VoxPopuli-Cleaned-AA — earlier subset overlap unresolved | 3.07 | 2620.26 |
| **Public-eight aggregate** | **4.90** | **5689.99** |

The complete run contains 74,443 eligible inputs from 74,544 source records after excluding 90 AMI and 11 GigaSpeech references under the official normalization rules. WER uses the pinned official normalizer and compound-aware alignment. The macro-average equally weights the eight per-dataset WERs rounded to two decimals.

Candidate results: https://huggingface.co/oruk/orukeet-r15-0100/blob/29e54a79807faada4b64e085a7b740ee78f05b89/evaluation/open_asr_20260919/public-eight-h200-results.json. Candidate Hub YAML: https://huggingface.co/oruk/orukeet-r15-0100/blob/29e54a79807faada4b64e085a7b740ee78f05b89/.eval_results/open_asr_leaderboard.yaml. The current pinned task registry has no Monsoon task ID; its measured value is retained in the all-eight companion summary and aggregate. A Monsoon task ID is not invented.

Historical r3 evidence remains in [the r3 model repository](https://huggingface.co/oruk/orukeet) and [its immutable evaluator Space revision](https://huggingface.co/spaces/oruk/open-asr-leaderboard-orukeet/tree/171ab5e79454cc3772a8700479cb150cfe4cc84f). Its scores and comparative placement do not apply to R15-0100. The default leaderboard additionally requires two private datasets; no private-set score, official default rank or maintainer acceptance is claimed here.
