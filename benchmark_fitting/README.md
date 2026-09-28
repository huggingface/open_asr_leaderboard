# Benchmark fitting

Two scorers for whether a model's output fits a benchmark's reference beyond what transcribing the audio explains. Both read the same published prediction manifests from [this HF Bucket](https://huggingface.co/buckets/hf-audio/asr_leaderboard_h200), and need no audio and no inference.

| file | |
| --- | --- |
| `score_voxpopuli_ref_errors.py` | section 1, with `ref_error_utils.py` |
| `score_ref_rendering.py` | section 2, with `ref_rendering_utils.py` |
| `show_flagged_spans.py` | inspects the individual spans section 2 scores |
| `utils.py` | shared by both scorers: locating, reading and naming manifests |

## 1. VoxPopuli reference error agreement

VoxPopuli's English references come from parliamentary records and sometimes
disagree with the audio. The leaderboard already reports WER against human
corrections for 628 clips from
[`ArtificialAnalysis/VoxPopuli-Cleaned-AA`](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA).

This scorer measures whether a model follows the original reference or the human
correction at those disagreements, of which 600 are detected between the splits `voxpopuli_test` and `voxpopuli_cleaned_aa_test` in [hf-audio/open-asr-leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard). Both references and model outputs are
normalized with this repository's `normalizer.EnglishTextNormalizer`.

Each disagreement receives one of three model verdicts:

- `ref`: every token in the original disagreement span survives;
- `consensus`: the original span does not survive and the corrected transcript is emitted;
- `excluded`: the output supports neither side, or the clip-level output is unusable.

Extra adjacent words do not erase evidence that the model reproduced the complete
original span. For boundary insertions, the corrected chunk must appear at the
same aligned boundary; extra words outside that chunk are allowed.

```
rate = ref / (ref + consensus)
```

A high rate can indicate exposure to the benchmark reference, but it is not WER
and does not by itself prove training-set contamination.

#### Method

The scorer diffs the original and corrected references, then evaluates each model
only at the resulting spans. Each contiguous span is one event regardless of its
word count. Replacements are recorded once as deletion runs. Correction
insertions are kept only at transcript boundaries, where their alignment is
unambiguous.

A disagreement or model verdict is excluded when:

- the references differ only by a small normalization artifact;
- the model reproduces fewer than `floor(0.5 × reference words)` tokens;
- the hypothesis is more than three times the reference length; or
- the model produces a third reading matching neither reference.

The output reports denominators and eligible clip counts, but not confidence
intervals, matching the leaderboard's WER and RTFx outputs. Several edits can
come from one clip, so the counts should not be treated as independent trials.

#### Usage

```bash
pip install -r requirements/requirements_jobs.txt

# Sync the public results bucket and score every model.
python benchmark_fitting/score_voxpopuli_ref_errors.py --bucket hf-audio/asr_leaderboard_h200

# Or score an existing download.
python benchmark_fitting/score_voxpopuli_ref_errors.py --preds_dir results

# Score a single model, by Hub id, bucket id, or a unique substring of either.
python benchmark_fitting/score_voxpopuli_ref_errors.py --preds_dir results --model openai/whisper-large-v3
```

Manifest filenames cannot carry the `/` of a Hub id, so the bucket writes
`openai-whisper-large-v3`; `--model` accepts either spelling and ignores case.
Both scorers here report the Hub id, restored by `normalizer.to_hub_ids`. That
inverse is lossy on its own — nothing in `abr-ai-niagara-19m-batch.en` says
whether the org is `abr` or `abr-ai` — so `normalizer/model_ids.py` keeps a set
of hyphenated orgs and a table of manifests never named after a Hub id at all
(`omniASR_CTC_7B_v2`, `stt_en_conformer_transducer_small`). A model from a new
hyphenated org needs an entry there, or it will be reported under a truncated
org name; two names colliding on one Hub id is reported and falls back to the
bucket names, rather than silently merging two models' rows.

With `--model`, the scorer prints that model's CSV line to stdout and writes no
files, so a single-model run cannot overwrite a full run's outputs. The row is
identical to the one a full run produces: the disagreement spans come from
diffing the two references, and each verdict depends only on that model's own
hypothesis.

The scorer needs prediction manifests for both `voxpopuli_test` and
`voxpopuli_cleaned_aa_test`. It reads only text manifests; no audio or inference is required.

#### Outputs

`ref_error_agreement_voxpopuli.csv` contains one row per model:

| column | meaning |
| --- | --- |
| `model` | Hub id, e.g. `microsoft/Phi-4-multimodal-instruct` |
| `rate` | `n_ref / n_eligible` |
| `n_ref` | spans matching the original reference |
| `n_eligible` | spans matching either reference, namely (ref + consensus) above |
| `n_clips` | clips contributing eligible spans |
| `wer_official` | WER % against the original reference |
| `wer_corrected` | WER % against the human correction |
| `n_wer_clips` | clips both WERs are computed over |

The two WERs are computed over one and the same clip set — every clip the model
has a hypothesis for and both references cover — with `kaldialign.batch_error_rate`
and `merge_compounds=True`, the call the leaderboard's own WER is computed with.
They therefore differ only in which reference they score against, so
`wer_official - wer_corrected` is the WER penalty the original reference's errors
impose on that model. It is an aggregate over the 628-clip subset, not comparable
to the leaderboard's `voxpopuli_test` column, which covers the full split.

`edits_voxpopuli.jsonl` records every disagreement with its clip key, location,
original and corrected spans, distance, and per-model verdicts so results can be
audited directly.

To score another model, place its raw prediction manifest alongside the existing
manifests and rerun. Both bucket layouts are supported:

```
<preds-dir>/<model>/MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_<dataset>.jsonl
<preds-dir>/<dataset>/<model>.jsonl
```

#### Limitations

- The rate covers disagreement spans, not the full dataset.
- Do not interpret small differences as a precise model ranking; spans from the
  same clip are related and denominators vary by model.
- It applies only where a trustworthy corrected reference exists. VoxPopuli is
  currently the only leaderboard dataset with the required pairing.
- A high rate is evidence of reference agreement, not proof of how that agreement
  arose.

#### Attribution

Adapted from *Quantifying Benchmark Optimization in ASR Models*
([reference implementation](https://github.com/tlebryk/asr-benchmark-optimization),
Apache-2.0).

## 2. Reference rendering agreement

English WER normalization erases writing choices such as `colour`/`color`,
`Mr.`/`mister`, `T. V.`/`TV`, `twenty`/`20`, and `e-mail`/`email`. This metric
reports how often a model reproduces the reference's form after correctly
transcribing the underlying words.

```text
agreement = matching reference renderings / eligible occurrences
```

The candidate list is curated from the leaderboard normalizer's spelling and
compound rules. It includes acoustically equivalent spelling, initialism,
number, title, and compound forms; ambiguous or pronunciation-changing rewrites
are excluded. Case and edge punctuation are ignored because runners can impose
those styles.

The leaderboard's `EnglishTextNormalizer` is still used to align the surrounding
words. Results are reported per dataset with the eligible count and class
breakdown. This is a diagnostic of reference-convention agreement, not proof of
training-set contamination.

#### Usage

```bash
pip install -r requirements/requirements_jobs.txt

# Sync the public prediction bucket and score every available short-form set.
python benchmark_fitting/score_ref_rendering.py

# Or use an existing download.
python benchmark_fitting/score_ref_rendering.py --preds_dir results \
  --datasets voxpopuli_cleaned_aa_test

# Score a single model, by Hub id, bucket id, or a unique substring of either.
python benchmark_fitting/score_ref_rendering.py --preds_dir results \
  --model openai/whisper-large-v3
```

Manifest filenames cannot carry the `/` of a Hub id, so the bucket writes
`openai-whisper-large-v3`; `--model` accepts either spelling and ignores case.
The outputs report the Hub id, as in section 1. With `--model` the scorer prints that model's CSV line
for each dataset to stdout and writes no files, so a single-model run cannot
overwrite a full run's outputs.
Every model is scored independently against the reference, so the printed row is
identical to the one a full run writes.

Each `ref_rendering_<dataset>.csv` contains `rate`, `n`, and the same pair for
`spelling`, `initialism`, `number`, and `lexical`.

| column | meaning |
| --- | --- |
| `model` | Hub id, e.g. `microsoft/Phi-4-multimodal-instruct` |
| `rate` | `n_agreed / n`, over every eligible occurrence in the dataset. Empty when `n` is 0, which keeps "never had the chance" distinct from "never agreed" |
| `n` | eligible occurrences: reference spans carrying a candidate form where the model transcribed the same words in the same place and produced a form from the same family. An occurrence the model missed, misrecognized, or rendered outside the family is not counted either way |
| `<class>_rate` | the same ratio restricted to one class, `0.0` when that class has no eligible occurrence |
| `<class>_n` | eligible occurrences in that class; the four sum to `n` |

Rows are ordered by model name, case-insensitively, as in section 1; the run
also prints a top ten by `rate` to stdout. An empty `rate` sorts no differently
from any other — it is a model with no eligible occurrence, not a zero. The
classes are:

| class | erased distinction | example arms |
| --- | --- | --- |
| `spelling` | British and American spellings of the same word | `colour` / `color`, `organise` / `organize` |
| `initialism` | how letter sequences are punctuated and spaced | `T.V.` / `T. V.` / `T V` / `TV` |
| `number` | digits against words, for cardinals and ordinals 11–99 | `twenty` / `20`, `twentieth` / `20th` |
| `lexical` | curated compounds and abbreviations | `e-mail` / `email`, `Mr.` / `mister`, `etc.` / `et cetera` |

Each occurrence is scored against the reference's own arm, so `rate` is
agreement with the reference's convention, not a preference for any one form.
Counts are occurrences, not clips, and one clip can contribute several, so they
should not be treated as independent trials.

To inspect the raw examples:

```bash
python benchmark_fitting/show_flagged_spans.py --preds_dir results \
  --dataset ami_test --class initialism --limit 20
```

Add `--html spans.html` for a browsable report.

