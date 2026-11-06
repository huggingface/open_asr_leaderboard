# Urdu multilingual ASR benchmark (proposed)

This contribution registers `ur` as a multilingual scoring language. The first
supported dataset is the **official, unmodified FLEURS Urdu (`ur_pk`) test split**,
loaded directly from `google/fleurs`. It does not create a private subset and
does not reuse any training data. The test split must be run in full for
leaderboard reporting; `--limit` experiments are development checks only.

## Normalization policy

The same `UrduTextNormalizer` is applied to references and predictions.
The policy normalizes well-defined Unicode Arabic/Urdu letter equivalents,
removes vowel marks and decorative elongation, converts Arabic/Extended Arabic
numerals to ASCII digits, and standardizes whitespace/punctuation. It does not
collapse Bari Yeh (`ے`) into Yeh (`ی`), Do-Chashmi Heh (`ھ`) into Heh (`ہ`), or
strip base hamza letters. It intentionally does not convert numbers to spoken
words: Urdu number verbalization is context dependent, and the upstream
`num2words` multilingual fallback is not guaranteed to support Urdu. This is
an explicit limitation to review with Urdu annotators before expanding.

## Source and benchmark naming

Source dataset: `google/fleurs`, configuration `ur_pk`, split `test`, license
CC BY 4.0. Existing leaderboard manifests from `normalizer.write_manifest`
will include the substring `google-fleurs_ur_pk_test`. `ml_ur` detects
`_ur_pk_test`. The combined multilingual CSV displays `ur_fleurs`.

## Example HF Job

From the leaderboard checkout, with a valid HF token and an authorized results
bucket, use the upstream runner (this PR only changes the supported Whisper
submit script):

```bash
HF_TOKEN=... RESULTS_BUCKET=YOURNAME/asr_leaderboard_multilingual \
ONLY_LANGUAGES=ur ONLY_DATASETS=fleurs \
USE_LOCAL_NORMALIZER=1 USE_LOCAL_SCRIPT=1 \
bash transformers/submit_jobs_whisper_ml.sh
```

The first run requires HF Jobs access and enough HF credit to execute the
selected model(s). By default the Whisper script has two Urdu-capable models.
Output is NOT verified merely because the script submits successfully.

Then score results (with your actual bucket name):

```bash
python scripts/score_bucket_results.py \
  --bucket YOURNAME/asr_leaderboard_multilingual \
  --multilingual --language ur --recompute_scores
```

## Pre-merge checks

- `python -m pytest -q tests/test_urdu_normalizer.py`
- `python -m py_compile normalizer/urdu_normalizer.py normalizer/data_utils.py normalizer/eval_utils.py scripts/score_bucket_results.py transformers/run_eval_ml.py`
- `bash -n transformers/submit_jobs_whisper_ml.sh`
- Inspect a sample of held-out Urdu reference and hypothesis pairs before
  finalizing normalization policy, including numerals and hamza variants.
- Demonstrate complete FLEURS Urdu test-split coverage and a valid result
  manifest from at least one multilingual baseline.

## Separately: Zipformer model

Model integration should be a separate PR or follow-up commit. It requires a
sherpa-onnx adapter, exact pinned model/tag, tokenizer from the public source,
and a full test-split manifest. The released model has gated access conditions,
so the maintainers must be able to access it in unattended jobs or evaluate
using explicitly authorized HF credentials. The author's reported result on 299
samples is not an official leaderboard number.
