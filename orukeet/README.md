# Orukeet: English short-form evaluation

This submission evaluates the released [oruk/orukeet](https://huggingface.co/oruk/orukeet) r3 checkpoint on the eight public English datasets. **r3 trained on LibriSpeech test-other and used it for selection; Monsoon English also influenced earlier training-partition selection.** See [the disclosure](training-disclosure.md). We request maintainer guidance on how this exposure should be represented and whether a different checkpoint is required for inclusion.

The model has 627,008,134 parameters, a FastConformer encoder and TDT decoder, supports 25 languages, and uses CC BY-SA 4.0 weights. This submission covers English only.

## Reproduce

The [evaluation Space](https://huggingface.co/spaces/oruk/open-asr-leaderboard-orukeet/tree/171ab5e79454cc3772a8700479cb150cfe4cc84f) contains the full evaluator and unmodified pinned official normalizer, its exact upstream patch, tests and bundle checksums. The launcher below verifies the immutable Space commit and every bundled file before invocation.

Install `huggingface_hub>=1.11.0`, authenticate with a write token, and create a results bucket in your account. From the repository root:

```sh
# Inspect the eight jobs without launching them.
bash orukeet/submit_jobs.sh --namespace YOUR_NAMESPACE --bucket YOUR_NAMESPACE/YOUR_BUCKET

# Submit at most two bounded H200 jobs at once, and wait for completion.
bash orukeet/submit_jobs.sh --namespace YOUR_NAMESPACE --bucket YOUR_NAMESPACE/YOUR_BUCKET \
  --execute --receipt-directory ./orukeet-job-receipts
```

The fixed profile uses the official NeMo Docker image by digest, H200, NeMo 2.7.2, Torch 2.8.0, BF16, batch size 128, greedy-batch decoding and `max_symbols=10`. It preserves original audio boundaries and the official reference filtering. Batch size 128 follows the existing Parakeet profile; exhaustive batch-size or throughput tuning is not claimed. CUDA synchronization brackets the timed full transcription pass after up to four warmup batches.

All model and dataset revisions are pinned. The NeMo checkpoint is downloaded from model revision `555136b50265a132d4cea0d35560c26fc4f657ab` and checked against SHA-256 `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`. Audio is staged on local job storage; result manifests and completion receipts go to your bucket. Earnings22 chunk metadata is retained for six-session scoring.

After completion, download the eight successful run directories from your bucket's `oruk-orukeet/` prefix into `./orukeet-results`. Retain their original manifest filenames, include exactly one successful run per dataset, and exclude failed attempts or canaries. Score with the pinned official normalizer:

```sh
pip install -r requirements/requirements_jobs.txt
hf download oruk/open-asr-leaderboard-orukeet --repo-type space \
  --revision 171ab5e79454cc3772a8700479cb150cfe4cc84f \
  --include 'evaluator/**' --local-dir ./orukeet-source
python -c 'import sys; sys.path.insert(0, "./orukeet-source/evaluator"); from normalizer.eval_utils import score_results; score_results("./orukeet-results", model_id="oruk/orukeet", language="en", families=["public"])'
```

## Measured H200 results

| Dataset | WER (%) | H200 RTFx |
|:--|--:|--:|
| AMI-Cleaned | 8.68 | 3741.22 |
| Earnings22-Cleaned-AA | 5.74 | 2982.17 |
| GigaSpeech-Cleaned | 7.48 | 5346.49 |
| LibriSpeech test-clean | 1.47 | 4180.11 |
| LibriSpeech test-other (trained on) | 2.84 | 4159.42 |
| SPGISpeech | 3.38 | 6419.21 |
| Monsoon English (prior selection exposure) | 3.84 | 4835.62 |
| VoxPopuli-Cleaned-AA | 2.46 | 2488.16 |
| **Public-eight aggregate** | **4.49** | **5545.68** |

All 74,443 eligible records were evaluated. These are self-reported public-set measurements. [Complete results and hashes](https://huggingface.co/oruk/orukeet/blob/main/evaluation/open_asr_20260918/public-eight-h200-results.json) and [evaluation YAML](https://huggingface.co/oruk/orukeet/blob/main/.eval_results/open_asr_leaderboard.yaml) are published in the model repository. The current Hub task registry has no Monsoon task ID; its result is reported explicitly in the complete companion report, rather than assigned an invented ID. Private-set verification and inclusion remain the maintainers' decision.
