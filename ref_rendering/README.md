# Reference rendering agreement

English WER normalization erases writing choices such as `colour` vs `color`,
`T. V.` vs `TV`, and `forty seven` vs `47`. This scorer measures how often a
model reproduces the benchmark reference's choice at those spans.

The scorer uses this repository's `normalizer.EnglishTextNormalizer`, the same
normalizer used for English WER. It reports one rate per English short-form
dataset.

## Method

1. Align each raw reference with its normalized form and label every rewrite.
2. Keep only spelling, acronym, and number spans where the written alternatives
   are acoustically equivalent.
3. For each model, require its normalized hypothesis to contain the span's words.
   Otherwise the span is ineligible for that model.
4. Compare the raw reference and hypothesis renderings after case-folding and
   removing edge punctuation.

```
rate = agreeing eligible spans / eligible spans
```

Denominators differ by model and are always reported.

## Classes

| class | example | treatment |
| --- | --- | --- |
| `spelling` | `colour` → `color` | scored |
| `acronym` | `T. V.` → `tv` | scored |
| `number` | `forty seven` → `47` | scored |
| `abbrev` | `Mr.` → `mister` | dropped |
| `case`, `punct` | `So` → `so`, `yeah,` → `yeah` | dropped |
| `other`, `contraction`, `disfluency` | heterogeneous rewrites | dropped |

Case and punctuation are excluded because evaluation runners can impose those
styles. Abbreviations are excluded because the normalizer's table also expands
ambiguous fragments and homographs such as `st` → `saint` and `gen` → `general`.
`other` rewrites are too heterogeneous to interpret.

Spelling pairs are blocked when meaning determines the form, such as `programme`
vs `program` in the software sense. Numbers are limited to a bare cardinal or
ordinal from 11–99, expressed as exactly one spoken number; currency, percentages,
and normalization merge artifacts are excluded. Optional hyphenation is ignored,
so `fifty-two` and `fifty two` are the same words-side rendering.

`voxpopuli_cleaned_aa_test` is not scored: it is a corrected reference for a
subset of `voxpopuli_test` on the same audio, so comparing both would make a
reference-editing decision look like an independent dataset result.

## Reproduce

```bash
pip install -r requirements/requirements_jobs.txt

# Sync the public bucket and score all English short-form datasets.
python ref_rendering/score_ref_rendering.py --bucket hf-audio/asr_leaderboard_h200

# Or score one dataset from an existing download.
python ref_rendering/score_ref_rendering.py --preds_dir results --datasets voxpopuli_test
```

The scorer reads raw references and hypotheses from the published prediction
manifests. It requires no audio or inference.

## Outputs

Each `ref_rendering_<dataset>.csv` contains one row per model:

| column | meaning |
| --- | --- |
| `model` | model id from the results bucket |
| `rate`, `n` | agreement rate and eligible-span count |
| `<class>_rate`, `<class>_n` | per-class diagnostic breakdown |

Confidence intervals are omitted, matching the leaderboard's WER and RTFx
outputs. Counts remain important because eligible spans differ by model and
dataset.

To score another model, place its raw prediction manifest alongside the existing
manifests and rerun. Both bucket layouts are supported:

```
<preds-dir>/<model>/MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_<dataset>.jsonl
<preds-dir>/<dataset>/<model>.jsonl
```

## Audit individual spans

`show_flagged_spans.py` prints reference context, the normalized span, each
model's aligned raw output, and its verdict. Sampling is seeded.

```bash
python ref_rendering/show_flagged_spans.py --preds_dir results \
  --dataset voxpopuli_test --models model-a,model-b \
  --class spelling --limit 20 --seed 7

# Render the same audit as a browsable page.
python ref_rendering/show_flagged_spans.py --preds_dir results \
  --dataset ami_test --class acronym --limit 30 --seed 7 \
  --html spans_ami.html
```

## Limitations

- A fixed output convention can match a corpus without benchmark exposure.
  Comparing one model across datasets is more informative than small within-set
  rank differences.
- Detection is one-armed: only references that depart from the normalizer's
  canonical form are visible. Inverting the normalizer produced too many false
  positives and is not included.
- Class composition varies by dataset, so rates should be read with their
  denominator and per-class counts.
- This is a diagnostic signal, not proof of training-set contamination.
