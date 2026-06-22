# ARK-ASR Local Evaluation Runbook

This runbook records the working setup on this machine for evaluating ARK-ASR
models against `hf-audio/open-asr-leaderboard` without redoing environment or
dataset discovery.

## Machine State

- Repo: `/workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr`
- Python: `/usr/local/bin/python`
- Verified versions: Python 3.11.14, torch 2.9.1+cu128, transformers 5.10.2, datasets 5.0.0
- GPUs: 8 x NVIDIA GeForce RTX 4090, 49140 MiB each
- Dataset root: `/data/hf_datasets`
- Scratch root used for parallel jobs: `/data/tmp` if writable, otherwise `/tmp`
- Model evaluated in the latest run: `/data/yumu/model/trained_model/ark_asr_3b_opd`

If running from a restricted sandbox and CUDA fails with `cudaGetDeviceCount`
or `nvidia-smi` cannot talk to the driver, rerun the command outside the
sandbox. The same Python environment works.

## Important Dataset Rule

Use the official local Earnings22 parquet directory:

```bash
/data/hf_datasets/hf-audio__open-asr-leaderboard__earnings22_official/earnings22
```

Do not use this directory for leaderboard scoring:

```bash
/data/hf_datasets/distil-whisper__earnings22/chunked
```

That chunked directory has about 57k filtered samples and produced the bad
`earnings22` result in the first 3B run. The official HF leaderboard
`earnings22` on this machine has 2741 raw rows and 2737 filtered rows.

## Expected Local Data

Only pass directories that contain the intended split parquet files. Some
source directories also contain train files, so the run command below creates
split-only symlink directories under the run scratch directory.

| Task | Source parquet files | Filtered samples |
| --- | --- | ---: |
| `ami/test` | `/data/hf_datasets/edinburghcstr__ami/ihm/test-*.parquet` | 11653 |
| `earnings22/test` | `/data/hf_datasets/hf-audio__open-asr-leaderboard__earnings22_official/earnings22/*.parquet` | 2737 |
| `gigaspeech/test` | `/data/hf_datasets/gigaspeech_parquet/parquet-data/test/*.parquet` | 19898 |
| `librispeech/test.clean` | `/data/hf_datasets/clean/test/*.parquet` | 2620 |
| `librispeech/test.other` | `/data/hf_datasets/other/test/*.parquet` | 2939 |
| `spgispeech/test` | `/data/hf_datasets/kensho__spgispeech/test/*.parquet` | 39341 |
| `voxpopuli/test` | `/data/hf_datasets/facebook__voxpopuli/en/test-*.parquet` | 1842 |

Expected final total after filtering: 81030 result rows.

To recheck counts without running the model:

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
PYTHONPATH=. python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
from normalizer import data_utils

items = {
    "ami": ("/data/hf_datasets/edinburghcstr__ami/ihm", "test-*.parquet"),
    "earnings22": ("/data/hf_datasets/hf-audio__open-asr-leaderboard__earnings22_official/earnings22", "*.parquet"),
    "gigaspeech": ("/data/hf_datasets/gigaspeech_parquet/parquet-data/test", "*.parquet"),
    "librispeech_clean": ("/data/hf_datasets/clean/test", "*.parquet"),
    "librispeech_other": ("/data/hf_datasets/other/test", "*.parquet"),
    "spgispeech": ("/data/hf_datasets/kensho__spgispeech/test", "*.parquet"),
    "voxpopuli": ("/data/hf_datasets/facebook__voxpopuli/en", "test-*.parquet"),
}

for name, (root, pattern) in items.items():
    raw = kept = files = 0
    for path in sorted(Path(root).glob(pattern)):
        files += 1
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            rows = pf.read_row_group(row_group).to_pylist()
            raw += len(rows)
            for row in rows:
                norm = data_utils.normalizer(data_utils.get_text(row))
                kept += int(data_utils.is_target_text_in_range(norm))
    print(f"{name}: files={files} raw={raw} kept={kept}")
PY
```

## One Command Full Run

Prefer the checked-in wrapper. It launches all seven leaderboard tasks in
parallel, creates split-only parquet symlink directories, checks task exit
codes, verifies result row counts, and runs `score_results`.

Run a preflight first if the machine has not been touched recently. This checks
the model path, parquet shards, and filtered sample counts without loading the
model or using GPUs:

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
MODEL_ID=/data/yumu/model/trained_model/ark_asr_3b_opd CHECK_ONLY=1 bash ark_asr/run_local_parallel_eval.sh
```

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
MODEL_ID=/data/yumu/model/trained_model/ark_asr_3b_opd bash ark_asr/run_local_parallel_eval.sh
```

By default the wrapper uses GPUs `0 1 2 3 4 5 6`, leaving GPU 7 free. Override
with `GPU_IDS`, for example `GPU_IDS="0 1 2 3 4 5 6 7"` if you later add
sharding or another task. Useful overrides:

```bash
MODEL_ID=/path/to/model \
RUN_ID=my_eval_$(date -u +%Y%m%d_%H%M%S) \
GPU_IDS="0 1 2 3 4 5 6" \
BATCH_SIZE=64 \
bash ark_asr/run_local_parallel_eval.sh
```

## Progress And Failure Checks

List running jobs:

```bash
cat ark_asr/logs/current_local_parallel_eval_run.txt
RUN_ID=$(cat ark_asr/logs/current_local_parallel_eval_run.txt)
cat ark_asr/logs/$RUN_ID/pids.txt
ps -fp $(awk '{print $1}' ark_asr/logs/$RUN_ID/pids.txt)
```

Watch logs:

```bash
RUN_ID=$(cat ark_asr/logs/current_local_parallel_eval_run.txt)
tail -f ark_asr/logs/$RUN_ID/*.log
```

Check completed result files and row counts:

```bash
RUN_ID=$(cat ark_asr/logs/current_local_parallel_eval_run.txt)
FINAL_DIR=ark_asr/results.$RUN_ID
find "$FINAL_DIR" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' | sort
wc -l "$FINAL_DIR"/*.jsonl
```

A correct full run should have 7 jsonl files and these line counts:

```text
ami                 11653
earnings22           2737
gigaspeech          19898
librispeech clean    2620
librispeech other    2939
spgispeech          39341
voxpopuli            1842
total               81030
```

## Rerun One Dataset

Use this when only one task failed or needs correction. Example for official
`earnings22`:

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr

MODEL_ID=/data/yumu/model/trained_model/ark_asr_3b_opd
RUN_ID=ark_asr_3b_opd_earnings22_official_$(date -u +%Y%m%d_%H%M%S)
BASE=/tmp/open_asr_leaderboard_$RUN_ID
LOG_DIR="$PWD/ark_asr/logs/$RUN_ID"
mkdir -p "$BASE/job" "$LOG_DIR"
printf '%s\n' "$RUN_ID" > "$PWD/ark_asr/logs/current_ark_asr_3b_opd_earnings22_official_run.txt"

cd "$BASE/job" && \
HF_HOME=/data/hf_datasets/.cache/huggingface \
HF_DATASETS_CACHE=/data/hf_datasets \
HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr \
CUDA_VISIBLE_DEVICES=0 \
python /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr/ark_asr/run_eval.py \
    --model_id="$MODEL_ID" \
    --dataset_path=hf-audio/open-asr-leaderboard \
    --dataset=earnings22 \
    --split=test \
    --local_parquet_dir=/data/hf_datasets/hf-audio__open-asr-leaderboard__earnings22_official/earnings22 \
    --device=0 \
    --batch_size=64 \
    --skip_eval_samples=0 \
    --max_eval_samples=-1 \
    --max_new_tokens=256 \
    --warmup_steps=0 \
    --dtype=float16 \
    --attn_impl=sdpa \
    --audio_input=array \
    --audio_decode=soundfile \
    --force_clean_exit \
    > "$LOG_DIR/earnings22_official.log" 2>&1

tail -40 "$LOG_DIR/earnings22_official.log"
wc -l "$BASE/job/results"/*.jsonl
```

After a single-task rerun, copy the new jsonl into a new final directory
together with the previous good jsonl files. Do not overwrite the old final
directory unless that is explicitly desired.

## Score Existing Results

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
MODEL_ID=/data/yumu/model/trained_model/ark_asr_3b_opd
RESULT_DIR=ark_asr/results.ark_asr_3b_opd_20260622_052631.corrected_official_earnings22

PYTHONPATH=. python - <<PY
from normalizer.eval_utils import score_results
score_results("$RESULT_DIR", "$MODEL_ID")
PY
```

Known corrected result for `/data/yumu/model/trained_model/ark_asr_3b_opd`:

```text
AMI WER          8.91
Earnings22 WER   8.25
Gigaspeech WER   7.30
LS Clean WER     1.09
LS Other WER     2.41
SPGISpeech WER   2.49
Voxpopuli WER    5.48
Composite WER    5.13
Composite RTFx 197.14
```

The older directory `ark_asr/results.ark_asr_3b_opd_20260622_052631` contains
the wrong chunked Earnings22 result and should not be used for final reporting.

## 3B Leaderboard Submission Artifacts

Prepared public Hub artifacts:

```text
Model repo:      https://huggingface.co/AutoArk-AI/ARK-ASR-3B
Eval YAML:       https://huggingface.co/AutoArk-AI/ARK-ASR-3B/blob/main/.eval_results/open_asr_leaderboard.yaml
HF Jobs Space:   https://huggingface.co/spaces/AutoArk-AI/open-asr-leaderboard-ark-asr-3b
Raw JSONL data:  https://huggingface.co/datasets/AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results
```

Use the local proxy for Hub or GitHub operations from this machine:

```bash
export HTTP_PROXY="http://127.0.0.1:17890"
export HTTPS_PROXY="http://127.0.0.1:17890"
```

This machine does not currently have the `hf` or `huggingface-cli` binaries.
Use `huggingface_hub` Python APIs for uploads, or install `hf` only if a future
workflow specifically needs the CLI.

Verify the online artifacts without submitting any jobs:

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi(token="hf_...")
for repo_id, repo_type in [
    ("AutoArk-AI/ARK-ASR-3B", "model"),
    ("AutoArk-AI/open-asr-leaderboard-ark-asr-3b", "space"),
    ("AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results", "dataset"),
]:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(repo_type, repo_id, "private=", info.private, "sha=", info.sha)
    for path in api.list_repo_files(repo_id=repo_id, repo_type=repo_type):
        print(" ", path)
PY
```

Dry-run the HF Jobs submission plan. This prints the seven jobs and exits
without creating jobs:

```bash
cd /workspace/yiming/open_asr_leaderboard.worktrees.add-ark-asr
DRY_RUN=1 \
RESULTS_BUCKET="AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results" \
HF_TOKEN=hf_... \
bash ark_asr/submit_jobs_ark_asr_3b.sh
```

To pin the model to a specific Hub commit/ref in the HF Jobs command, set
`MODEL_REVISION`. This maps to `run_eval.py --revision`, matching the official
Transformers evaluation Space behavior:

```bash
MODEL_REVISION="<commit-or-ref>" \
DRY_RUN=1 \
RESULTS_BUCKET="AutoArk-AI/ark-asr-3b-open-asr-leaderboard-results" \
HF_TOKEN=hf_... \
bash ark_asr/submit_jobs_ark_asr_3b.sh
```

Do not run the non-dry-run command unless maintainers explicitly ask for an HF
Jobs rerun and a writable HF bucket has been chosen. The current submitted
numbers are local 8x RTX 4090 results, with raw JSONL manifests uploaded for
maintainer re-scoring or rerun.

## Code Assumption

The local parquet path mode depends on `ark_asr/run_eval.py` supporting parquet
rows with either an `audio` dict or top-level `bytes`/`path` columns. Keep the
`extract_audio_from_local_parquet_row` helper and the local parquet iterator
behavior when rebasing this branch.
