# VoiceCodeBench

[VoiceCodeBench](https://huggingface.co/datasets/besimple-ai/voice-code-bench) has 300 English workplace
recordings (35–123 s) with 1,482 annotated entities. Two metrics come from the same raw transcripts:

- **WER**, scored like every other dataset (same normalizer and scorer). It is part of the default
  short-form average (family `public` in `normalizer/eval_utils.py`).
- **CTEM** (Canonical Token/Entity Match): 100 × entities correctly recovered / all entities, pooled over
  recordings. Higher is better and it is never averaged with WERs.

## Running it

VoiceCodeBench is a normal dataset entry in the submit scripts (`nemo_asr/submit_jobs_parakeet.sh`,
`transformers/submit_jobs_whisper.sh`, `transformers/submit_jobs_qwen3asr.sh`, `sov_transcribe/submit_jobs.sh`),
wired through the helpers in `scripts/submit_utils.sh`. A script lists
`"voice_code_bench test besimple-ai/voice-code-bench"` and calls `vcb_init`, `vcb_dataset_config`,
`vcb_track_model` and `vcb_grade`. To add the dataset to another script, do the same.

```bash
HF_TOKEN=hf_... bash nemo_asr/submit_jobs_parakeet.sh                                   # everything
HF_TOKEN=hf_... ONLY_DATASETS="voice_code_bench" bash nemo_asr/submit_jobs_parakeet.sh  # just VoiceCodeBench
```

Transcripts go to the model's folder in the results bucket (`hf-audio/asr_leaderboard_h200`) like the other
datasets. When all models are done, one grading job per model runs at once (about 13 minutes and $1 each on an
H200), then each model's CTEM and WER are printed and a `<manifest>.summary.json` is stored beside the
transcripts. Grading also writes `<manifest>.ctem.json` and `<manifest>.qwen27b-verifier-cache.json` there.

Options:

- `VCB_SCORING=0` only transcribes. Grade later with `vcb_submit_grading <model-folder>/<manifest>.jsonl`
  (after `source scripts/submit_utils.sh` with `RESULTS_BUCKET` set).
- `PUBLISH_RESULTS=0` skips the summary upload.
- `VCB_JUDGE=fp8` grades with the FP8 judge (see below).
- `VERIFIER_MODE=replay` re-scores from a complete cache on a CPU job. `ALLOW_PARTIAL=1` is for smoke tests.
- `SCORING_FLAVOR`, `SCORING_IMAGE`, `SCORING_TIMEOUT`, `ORG_NAME` change the grading job. Keep the pinned model
  and vLLM configuration when comparing scores.

A model whose transcription failed in this run is reported and skipped, never graded from an old manifest.
Run only one grading job per manifest at a time.

Model-specific notes: every clip is longer than 30 s, so Whisper uses long-form decoding (`--longform`, batch 32)
and Qwen3-ASR batch 64 for this dataset. Models with a shorter input limit need the same care.
The scripts pass `normalizer/` and `run_eval.py` as one command argument, which Linux limits to 131072 bytes, so
`local_normalizer_inject` leaves out the judge module and prompt (only the grading job needs them).
Watch the margin when adding files to `normalizer/`.

## Scoring protocol

The judge is the Apache-2.0 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint, pinned to
revision `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`, served by a loopback vLLM
(`vllm/vllm-openai:v0.28.0-cu129`, BF16, 32,768-token context, `qwen3` reasoning parser). There is one call per
transcript, with the full text, lossless 256-character segments and all targets. A correct entity must be clearly
derivable from the transcript. Gold metadata never fills in missing content. Requests use temperature zero,
`xhigh` reasoning effort, an 8,192-token thinking budget and a 16,384-token completion limit.

The [system prompt](normalizer/voice_code_bench_prompt.txt), strict response schema and citation validation are in
[`normalizer/voice_code_bench_judge.py`](normalizer/voice_code_bench_judge.py). Any malformed, truncated, wrong-model
or failed response stops scoring (no repair, no automatic retry, no hosted API). Validated responses are checkpointed
to the bucket, so resuming reuses them. Caches are bound to the model, runtime, implementation and prompt
(`config_digest`) and incompatible ones fail.

Up to 64 requests run concurrently (`RUNTIME["max_num_seqs"]`, also vLLM's `--max-num-seqs`; `--verifier-workers`
overrides the request count). Compared with 16, this makes a grading job about 1.8× faster and 44% cheaper.
128 was no faster and brought requests close to the 600 s timeout.

Annotations, audio and the metric are pinned to dataset revision `bef2824f83ef1c796f3e79731a3b0741708730df`.
Empty transcripts stay in the denominator. Complete coverage is required unless `--allow-partial` marks a smoke test.
Historical GPT scores must not be relabeled as scores from this protocol.

**Validation.** On a frozen benchmark of 693 manually assessed entity decisions the judge scored 629/693 (the earlier
GPT judge: 597/693) and passed 70/70 synthetic controls. These figures were measured before rule 11 (spoken zero:
a standalone "O"/"oh" between digit tokens counts as 0 when the target is numeric) was added to the prompt and the
default concurrency raised to 64. Identical runs differ by about 5 of 1,482 verdicts (±0.1–0.2 CTEM).

## FP8 judge

`VCB_JUDGE=fp8` grades with `Qwen/Qwen3.8-27B-FP8` (pinned in `scripts/score_voice_code_bench_job.py`). It has its
own judge id and writes `<manifest>.ctem-fp8.json` and `<manifest>.qwen27b-fp8-verifier-cache.json` beside the BF16
files. No summary is published. Over four models it agreed with BF16 on 99.5% of verdicts, ranked them identically
and was about 9% cheaper, so scores are close but not interchangeable. BF16 is the default.

## Baseline and license

The dataset's [published Parakeet predictions](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/baselines/predictions/modal_nvidia_parakeet_tdt_0_6b_v3.json)
score 1,165/1,482 = 78.61% CTEM under the historical GPT verifier. The dataset is test-only. Non-audio materials use
MIT and audio also prohibits voice cloning; see its
[license](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/bef2824f83ef1c796f3e79731a3b0741708730df/LICENSE).
