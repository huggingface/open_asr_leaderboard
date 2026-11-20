# SpeedASR-v1-65M

Offline English Zipformer transducer with **65,551,059 parameters**. Inference uses FP32, modified beam search with four active paths, context size three, and the same VoiceA text postprocessing for every dataset. No external language model, prompt, domain hint or dataset-specific vocabulary is used.

- [Public model and inference package](https://huggingface.co/SpeedAudio261119/SpeedASR-v1-65M/tree/a6f5cacea359929b52ae41b77f6f634441cf0546)
- [Evaluation Docker Space](https://huggingface.co/spaces/SpeedAudio/open-asr-leaderboard-speedasr)
- [Batched `run_eval.py` adapter](https://huggingface.co/SpeedAudio261119/SpeedASR-v1-65M/blob/a6f5cacea359929b52ae41b77f6f634441cf0546/leaderboard/run_eval.py)
- [Results bucket](https://huggingface.co/buckets/SpeedAudio/speedasr-open-asr-results)

The Space contains the complete runtime; this directory supplies its HF Jobs launcher. The adapter uses the official `normalizer/data_utils.py` for data loading and manifest writing, and saves raw references/predictions plus Earnings22 parent/chunk metadata. The container pins the official scorer to `4a113dd90c9ffd011d7d3d105da76980d511d029`. Checkpoint, tokenizer and configuration are downloaded from the immutable model revision above.

## Run the public English short-form track

Use Bash and Python 3.10+ with `huggingface_hub==1.18.0`, an HF write token in the process environment, available Jobs credits, a built Space and an existing writable bucket:

```bash
python -m pip install 'huggingface_hub==1.18.0'
# Set HF_TOKEN in the process environment; do not put it in a script or commit.
bash speedasr/submit_jobs.sh
```

The launcher defaults to the Space and bucket linked above. It starts **one H200 job at a time**, with a **one-hour runtime limit per job**, and explicitly checks terminal job status before starting the next dataset. Failed or canceled jobs stop the run. `hf jobs run` streaming its logs does not itself establish successful job completion in Hub CLI 1.18.0.

The default selection contains AMI Cleaned, GigaSpeech Cleaned, VoxPopuli Cleaned AA, Earnings22 Cleaned AA Chunked, SPGISpeech, URGENT2024 and Monsoon English. `INCLUDE_URGENT_CLEAN=1` adds the optional clean URGENT set; it remains outside the seven-set public mean. Earnings22 is evaluated in full and scored at parent-session level.

```bash
# Evaluate one complete dataset first.
ONLY_DATASETS=ami_cleaned bash speedasr/submit_jobs.sh

# Override throughput limits after profiling on H200; decoder settings stay fixed.
BATCH_SIZE=4096 BATCH_AUDIO_SECONDS=10000 \
  bash speedasr/submit_jobs.sh

# Optional organization billing or a different writable bucket.
ORG_NAME=my-org RESULTS_BUCKET=my-org/asr-results \
  bash speedasr/submit_jobs.sh
```

`BATCH_SIZE=2048`, `BATCH_AUDIO_SECONDS=5000` and `TORCH_THREADS=16` are starting throughput settings. They are not a claim that the maximum practical H200 batch size has been measured. Use the maximum practical batch settings for the final leaderboard throughput report and retain the generated metadata. Five fixed ten-second synthetic warmup inputs are independent of evaluation data. Audio loading/resampling, warmup and scoring are excluded from timed inference; feature extraction, transfers, decoding and text postprocessing are included.

Each run gets a unique UTC `RUN_ID`. Jobs persist JSONL manifests, metadata and logs under `<model-folder>/<RUN_ID>/<dataset>` in the bucket. The launcher saves job IDs/statuses and downloads that run under `speedasr/cloud_runs/<RUN_ID>/`. It does not change the model repository or open another pull request.

The launcher records the model pin, Space source commit and available Job image metadata, and stops if the Space source/build changes between datasets. `SPACE_REVISION=<full-Space-commit-SHA>` can require a particular source commit. Hub 1.18.0's documented Space-image shorthand has no separate image-revision parameter; a recorded source SHA is not an OCI image-digest pin.

The package also contains independently checked RTX 3090 Ti local results: seven-public-set mean WER **6.9043%** and pooled RTFx **192.6293**. Those are local measurements, not H200 results. Public cloud results and private-set evaluation must be reported separately; private sets are evaluated by the leaderboard maintainers.

Model metadata: Apache-2.0; 0.065551059B parameters; one language (English); Zipformer encoder; stateless transducer decoder. See the model card for the disclosed training information and limitations.
