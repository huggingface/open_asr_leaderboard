# SoundsGoodAI Fast GPU ASR

Evaluate Zipformer and Parakeet using [fast-gpu-asr](https://pypi.org/project/fast-gpu-asr/):
TensorRT encoders, GPU-native audio features, and batched GPU decoding.
The [NeMo runners](../nemo_asr/) remain available for upstream comparisons.

## Models

| Checkpoint | Decoder | Reported as |
| --- | --- | --- |
| `soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M` | RNN-T | `soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M transducer_modified_beam_search` |
| `soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M` | CTC | `soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M ctc_greedy_search` |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT | `nvidia/parakeet-tdt-0.6b-v3 (fast-gpu-asr)` |
| `nvidia/parakeet-tdt-0.6b-v2` | TDT | `nvidia/parakeet-tdt-0.6b-v2 (fast-gpu-asr)` |
| `nvidia/parakeet-ctc-0.6b` | CTC | `nvidia/parakeet-ctc-0.6b (fast-gpu-asr)` |
| `nvidia/parakeet-ctc-1.1b` | CTC | `nvidia/parakeet-ctc-1.1b (fast-gpu-asr)` |

Zipformer RNN-T and CTC are separate configurations. Parakeet RNN-T is not
supported; export rejects incompatible checkpoint architectures.

The Parakeet checkpoints are also evaluated with the [NeMo runners](../nemo_asr/),
so their results here carry a `(fast-gpu-asr)` suffix to keep the two backends
apart in manifests and scoring. The suffix is the last `MODEL_CONFIGS` field and
is passed to `run_eval.py` as `--report-name`; `--model-id` stays the Hub
repository used for download, export, and engine caching. The reported name also
names the result folder, slugified (`nvidia-parakeet-tdt-0.6b-v3-fast-gpu-asr`),
so it must be unique across `MODEL_CONFIGS`: the two Zipformer rows differ only
by decoder and use it as their suffix.

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
repository, family, checkpoint, decoder, beam, batch size, and optional
reported-name suffix; `DATASET_CONFIGS` selects datasets; `COMMON_ARGS` sets
precision, duration profiles, and warm-ups.

```bash
bash soundsgoodai/run_models.sh
```

Defaults are **FP16, batch 256**, beam **10** for Zipformer RNN-T, **6** for
Parakeet TDT V2/V3, and **1** for CTC, a **0.1 / 10 / 40-second** duration profile
(min/opt/max), optimization level **5**, and blank penalty **0.0**. The eight
dataset splits are LibriSpeech clean/other, AMI, chunked Earnings22, GigaSpeech, SPGISpeech,
VoxPopuli, and Monsoon English. Reduce batch sizes if needed for GPU memory.

Local suites save to `soundsgoodai/runs/<RUN_ID>/<reported-name>/results/`.
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

- `PARALLEL_DATASETS=1` in `config.sh` submits one job per dataset instead and
  runs a model's jobs in parallel, so its wall clock is the slowest dataset
  rather than their sum. Each job then builds its own engines, so the GPU cost
  is one export per dataset. Models are still handled one at a time, each scored
  before the next is submitted.
- `FLAVOR`, `ORG_NAME`, and `TIMEOUT` (default `8h`) control scheduling.
- `SPACE` selects the image; `RESULTS_BUCKET` must name a bucket you can write to.
- `ONLY_DATASETS="librispeech spgispeech"` selects clean, other, and SPGISpeech.
- Local `run_eval.py` and `normalizer/` are injected into each job by default, so
  runner and normalizer changes take effect without updating the Space. Set
  `USE_LOCAL_SCRIPT=0` or `USE_LOCAL_NORMALIZER=0` to use the image's versions.
  Dependency changes still require an image rebuild.

`HF_TOKEN` is passed as a job secret. Each completed dataset's transcripts,
metadata, and log are saved under
`hf://buckets/<RESULTS_BUCKET>/<reported-name>/`, surviving later
evaluation failures. Once a job ends, the launcher downloads results to
`soundsgoodai/results/<RUN_ID>/<reported-name>/`, checks manifest counts and
metadata sidecars, and prints the per-model summary, also written to
`scores.log`. The job's own output (export and per-dataset progress) is not
echoed to the terminal; read `<reported-name>/job.log` for it, or
`job-<dataset>-<split>.log` with `PARALLEL_DATASETS=1`. A failed job or a
missing dataset is a warning, not an error: the summary covers whatever arrived
and the remaining models still run.
`RUN_ID` defaults to a UTC timestamp-based name and groups local results only;
existing local run directories are rejected. Bucket folders are keyed on the
reported name alone, so rerunning a configuration overwrites its manifests
there, and manifests from a previous run over different datasets are downloaded
alongside the new ones.

## Multilingual

`nvidia/parakeet-tdt-0.6b-v3` is the only multilingual checkpoint here; V2, the
Parakeet CTC models, and Zipformer are English-only. Its own launcher,
[config_ml.sh](config_ml.sh), [run_eval_ml.py](run_eval_ml.py), and
[submit_jobs_ml.sh](submit_jobs_ml.sh), mirrors the English ones and evaluates
**FLEURS, Mozilla Common Voice, and Multilingual LibriSpeech** in the six
languages the leaderboard reports (`de`, `fr`, `it`, `es`, `pt`, `nl`), sixteen
dataset/language combinations from
[open-asr-leaderboard-multilingual-datasets](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-multilingual-datasets):

```bash
HF_TOKEN=hf_... bash soundsgoodai/submit_jobs_ml.sh
HF_TOKEN=hf_... ONLY_LANGUAGES="nl" bash soundsgoodai/submit_jobs_ml.sh
HF_TOKEN=hf_... ONLY_DATASETS="fleurs mcv" ONLY_LANGUAGES="nl de" bash soundsgoodai/submit_jobs_ml.sh
```

Dataset configuration names are `<dataset>_<language>`, such as `fleurs_de`. The
launcher takes the language from that suffix for `ONLY_LANGUAGES`, for
`--language`, and for the summaries, and `run_eval_ml.py` derives it the same
way when `--language` is absent. `ONLY_DATASETS` accepts either the full
configuration name (`mls_pt`) or the dataset alone (`mls`).

`config_ml.sh` sources `config.sh`, so decoders, precision, warm-ups, and
`PARALLEL_DATASETS` are shared; it replaces the models, the datasets, the
duration profile, and the batch size. The profile goes to **60 seconds**
because FLEURS reaches 53 and overlong clips fail rather than being truncated,
and the batch size drops to **128** to pay for it: the feature plugin stages
one TensorRT workspace proportional to `batch_size * max_audio_seconds`, the
English 256 x 40 s already uses 98.8% of its signed 32-bit limit, and exceeding
it fails the export before any audio is read. At 60 seconds the ceiling is 172.
Raising either one means lowering the other. `RESULTS_BUCKET` defaults to
`hf-audio/asr_leaderboard_multilingual`, so multilingual manifests never share a
bucket folder with the English ones despite the identical reported name.
Everything else, injection of the local runner and normalizer, bucket sync,
manifest and metadata checks, and the `RUN_ID` layout, works as described above.

Scoring uses `ml_normalizer` for the evaluated language rather than the English
normalizer: it also spells out digits, so `tien uur` and `10 uur` agree. Compound
word boundaries are aligned before the compound-aware WER. `submit_jobs_ml.sh`
calls `score_results` once per language, each restricted to its own `ml_<lang>`
family, so every summary is normalized for the language it covers.

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
