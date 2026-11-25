# SpeedASR-v1-65M

Offline English Zipformer transducer with **65,551,059 parameters**. Inference uses FP32, modified beam search with four active paths, context size three, and the same VoiceA text postprocessing for every dataset. No external language model, prompt, domain hint or dataset-specific vocabulary is used.

- [Public model and inference package](https://huggingface.co/SpeedAudio261119/SpeedASR-v1-65M/tree/a6f5cacea359929b52ae41b77f6f634441cf0546)
- [Evaluation Docker Space](https://huggingface.co/spaces/SpeedAudio/open-asr-leaderboard-speedasr)
- [Batched evaluation entry point](run_eval.py)
- [Results bucket](https://huggingface.co/buckets/SpeedAudio/speedasr-open-asr-results)

The Space supplies the model runtime and pins the official scorer to `4a113dd90c9ffd011d7d3d105da76980d511d029`. This directory contains the evaluator and the HF Jobs launcher. The evaluator uses the official `normalizer/data_utils.py` for data loading and manifest writing, and saves raw references/predictions plus Earnings22 parent/chunk metadata. The checkpoint, tokenizer and configuration come from the immutable model revision above.

## Run the public English short-form track

Use Bash and Python 3.10+ with `huggingface_hub==1.18.0`, an HF write token in the process environment, available Jobs credits, a built Space and an existing writable bucket:

```bash
python -m pip install 'huggingface_hub==1.18.0'
# Set HF_TOKEN in the process environment; do not put it in a script or commit.
bash speedasr/submit_jobs.sh
```

By default, the launcher copies the checked-in `run_eval.py` to `/app/leaderboard/run_eval.py` in the job, using the repository's shared launcher helper. Set `USE_LOCAL_SCRIPT=0` to use the evaluator bundled in the Space.

The launcher defaults to the Space and bucket linked above. It uses the Hugging Face SDK to submit **one H200 job at a time** with a **one-hour runtime limit per job**. Each job must save its prediction manifest and metadata and reach `COMPLETED` before the launcher continues to the next dataset.

The default selection contains AMI Cleaned, GigaSpeech Cleaned, VoxPopuli Cleaned AA, Earnings22 Cleaned AA Chunked, SPGISpeech, URGENT2024 and Monsoon English. `INCLUDE_URGENT_CLEAN=1` adds the optional clean URGENT set; it remains outside the seven-set public mean. Earnings22 is evaluated in full and scored at parent-session level.

```bash
# Evaluate one complete dataset first.
ONLY_DATASETS=ami_cleaned bash speedasr/submit_jobs.sh

# Use the same limits verified by the H200 pilot; decoder settings stay fixed.
BATCH_SIZE=2048 BATCH_AUDIO_SECONDS=5000 \
  bash speedasr/submit_jobs.sh

# Optional organization billing or a different writable bucket.
ORG_NAME=my-org RESULTS_BUCKET=my-org/asr-results \
  bash speedasr/submit_jobs.sh
```

`BATCH_SIZE=2048` and `BATCH_AUDIO_SECONDS=5000` are the defaults because that setting was fastest in the H200 pilot. The evaluator uses 16 CPU threads and warms up with five fixed ten-second synthetic inputs. Timed inference includes feature extraction, transfers, decoding and text postprocessing. Audio loading, resampling, warmup and scoring run outside the timed interval.

## H200 performance pilot — 21 November 2026

One NVIDIA H200 ran FP32 inference with beam 4 on **2,304 speech clips**, totaling 20,055.47 audio seconds per configuration. The workload crops, concatenates and reuses 32 excerpts from eight American and British recordings in the [Open Speech Repository](https://www.voiptroubleshooter.com/open_speech/about.html). The pilot measures throughput on speech independent of the Open ASR testsets.

All configurations used a cap of 2,048 clips. The audio budget is the longest clip's duration multiplied by the number of clips in the batch.

| Padded-audio budget (seconds) | Largest observed batch | RTFx | Peak allocated GPU memory (GiB) |
|---:|---:|---:|---:|
| **5,000 — retained default** | 625 | **209.13** | 20.676 |
| 9,000 | 1,070 | 205.32 | 36.930 |
| 14,000 | 1,174 | 203.68 | 57.194 |

The pilot measured one pass per configuration. The 5,000-second budget was fastest across the three completed trials.

The 20,000-second trial hit PyTorch's `canUse32BitIndexMath(input) && canUse32BitIndexMath(output)` limit in a depthwise convolution. The job ended after 5 minutes 14 seconds of running time. The current runtime retries that error and CUDA out-of-memory errors with smaller batches when possible. It applies the audio budget to long recordings and records post-warmup retries under `cuda_batch_retries`. Isolated regression tests exercised both retry paths.

The pilot used model revision `a6f5cacea359929b52ae41b77f6f634441cf0546` and Space source `2d0cadf1ee431f4a07e2ec2fb53b39c24df269be`. Its RTFx must be kept separate from the seven-testset WER and from a future full H200 leaderboard run.

Each run gets a unique UTC `RUN_ID`. Jobs persist JSONL manifests, metadata and logs under `<model-folder>/<RUN_ID>/<dataset>` in the bucket. The launcher saves job IDs/statuses and downloads that run under `speedasr/cloud_runs/<RUN_ID>/`. It does not change the model repository or open another pull request.

The launcher records the model pin, Space source commit and available Job image metadata, and stops if the Space source/build changes between datasets. `SPACE_REVISION=<full-Space-commit-SHA>` can require a particular source commit. Hub 1.18.0's documented Space-image shorthand has no separate image-revision parameter; a recorded source SHA is not an OCI image-digest pin.

The [official-filtered local rescore](https://huggingface.co/SpeedAudio261119/SpeedASR-v1-65M/blob/6ca2db465b582b13944ceb6276561a349efb43ed/benchmark_results/local_3090ti_20261119/official_filter_rescore.json) reports a seven-public-set mean WER of **6.8871%** over **69,884 clips** from saved RTX 3090 Ti predictions. The original local inference run recorded **192.6293 RTFx**. Report public H200 results separately. The leaderboard maintainers evaluate the private sets.

Model metadata: Apache-2.0; 0.065551059B parameters; one language (English); Zipformer encoder; stateless transducer decoder. See the model card for the disclosed training information and limitations.
