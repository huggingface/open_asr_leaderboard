# SoundsGoodAI Fast GPU ASR

Evaluate Zipformer and Parakeet using [fast-gpu-asr](https://pypi.org/project/fast-gpu-asr/):
TensorRT encoders, GPU-native audio features, and batched GPU decoding.
The [NeMo runners](../nemo_asr/) remain available for upstream comparisons.

## Models

| Checkpoint | Decoder |
| --- | --- |
| `soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M` | RNN-T; CTC |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT |
| `nvidia/parakeet-tdt-0.6b-v2` | TDT |
| `nvidia/parakeet-ctc-0.6b` | CTC |
| `nvidia/parakeet-ctc-1.1b` | CTC |

Zipformer RNN-T and CTC are separate configurations. Parakeet RNN-T is not
supported; export rejects incompatible checkpoint architectures.

## Local Setup

Use Linux x86-64, **Python 3.12-3.14**, a Turing (SM75) or newer NVIDIA GPU,
[driver 580 or newer](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
and a CUDA-compatible `g++` for decoder-kernel compilation. BF16 requires SM80
or newer. From the repository root, install CPU-only PyTorch first:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/requirements_zipformer.txt
pip check
hf auth login
```

The wheel bundles TensorRT plugins, and dependencies provide CUDA/TensorRT.
SoundFile handles audio loading. No NeMo, Icefall, k2, TorchCodec, or separate
system CUDA toolkit is needed; the requirements file above is sufficient.

Both launchers share [config.sh](config.sh): `MODEL_CONFIGS` sets each model's
repository, family, checkpoint, decoder, beam, and batch size; `DATASET_CONFIGS`
selects datasets; `COMMON_ARGS` sets precision, duration profiles, and warm-ups.

```bash
bash soundsgoodai/run_models.sh
```

Defaults are **FP16, batch 256**, beam **6** for Zipformer RNN-T and **1** for
Parakeet and CTC, a **0.1 / 10 / 40-second** duration profile (min/opt/max),
optimization level **5**, and blank penalty **0.0**. The eight dataset splits are
LibriSpeech clean/other, AMI, chunked Earnings22, GigaSpeech, SPGISpeech,
VoxPopuli, and Monsoon English. Reduce batch sizes if needed for GPU memory.

Local suites save to `soundsgoodai/runs/<RUN_ID>/<model>/<decoder>/results/`.
Engines are cached in `soundsgoodai/engines/`, overridable with `ENGINE_CACHE`.
Clear the cache after TensorRT or plugin upgrades: package versions are not
part of its key. Select the GPU with `--device` in `COMMON_ARGS`, without
remapping the parent `CUDA_VISIBLE_DEVICES`.

For a three-clip Zipformer CTC smoke test:

```bash
PYTHONPATH=. python soundsgoodai/run_eval.py \
  --model-id soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M \
  --model-family zipformer --checkpoint-file model.pt \
  --decoder-type ctc_greedy_search --batch-size 2 --beam 1 \
  --precision fp16 --dataset librispeech --split test.clean \
  --min-audio-seconds 0.1 --opt-audio-seconds 10 --max-audio-seconds 40 \
  --max-eval-samples 3
```

Direct runs write to `./results`; keep different configurations in separate
working directories. Use `--max-eval-samples=-1` for full evaluations.
Truncated chunked datasets are rejected because scoring needs complete parents.

## HF Jobs

**[HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) are paid.**
Update the [Space](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-zipformer)
with the runner, normalizer, and dependencies, wait for its build to succeed, and
review `config.sh` before submitting:

```bash
HF_TOKEN=hf_... bash soundsgoodai/submit_jobs.sh
```

The default matrix runs **six sequential jobs**, one per model/decoder, each
evaluating all eight splits on **one H200**. The first dataset builds engines
in `/app/engines`; subsequent datasets reuse them on the same GPU.

- `FLAVOR`, `ORG_NAME`, and `TIMEOUT` (default `8h`) control scheduling.
- `SPACE` selects the image; `RESULTS_BUCKET` must name a bucket you can write to.
- `ONLY_DATASETS="librispeech spgispeech"` selects clean, other, and SPGISpeech.
- `USE_LOCAL_SCRIPT=1` and `USE_LOCAL_NORMALIZER=1` inject local copies instead
  of the image's versions. Dependency changes still require an image rebuild.

`HF_TOKEN` is passed as a job secret. Each completed dataset's transcripts,
metadata, and log are saved under
`hf://buckets/<RESULTS_BUCKET>/<RUN_ID>/<model>/<decoder>/`, surviving later
evaluation failures. After a successful job, the launcher downloads results to
`soundsgoodai/results/<RUN_ID>/<model>/<decoder>/`, verifies manifest counts and
metadata sidecars, and writes aggregate scoring to `scores.log`.
`RUN_ID` defaults to a UTC timestamp-based name; existing local run directories
are rejected. Failed jobs are not scored as complete suites.

## Measurement and Scoring

- **Preparation:** use upstream reference filtering and mono 16 kHz float32 audio.
  Zipformer resamples through PCM16 `audioop.ratecv`; Parakeet uses `soxr_hq`.
  Sort by descending sample count, breaking ties by ascending utterance ID.
- **Execution:** five warm-ups on the first batch by default, then one measured
  pass, including the final partial batch. Inference results are never cached.
- **Timing:** synchronize the full `ASR` call, including staging, transfers,
  features, encoding, decoding, text, and timestamps. Exclude loading/export,
  file I/O, resampling, sorting, and scoring.
- **RTFx:** total real audio duration / total inference time. Padding does not
  count as audio. Batch time is shared equally among manifest rows, not measured
  as per-clip latency.
- **WER:** save raw references/predictions and audio IDs, reconstruct Earnings22
  parents in chunk order, then use the upstream compound-aware English scorer.

Checkpoints come from Hub `main`. Each JSONL manifest has a metadata sidecar with
the resolved checkpoint revision, export settings, GPU/CUDA details, and dataset
fingerprint. Retain the evaluation code revision, package versions, and logs for
reproducibility. Overlong clips fail rather than being truncated or omitted;
increase the duration profile and rerun the full configuration.
