# Reference rendering agreement

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

## Run

```bash
pip install -r requirements/requirements_jobs.txt

# Sync the public prediction bucket and score every available short-form set.
python ref_rendering/score_ref_rendering.py

# Or use an existing download.
python ref_rendering/score_ref_rendering.py --preds_dir results \
  --datasets voxpopuli_cleaned_aa_test

# Score a single model, by Hub id, bucket id, or a unique substring of either.
python ref_rendering/score_ref_rendering.py --preds_dir results \
  --model openai/whisper-large-v3
```

Manifest filenames cannot carry the `/` of a Hub id, so the bucket writes
`openai-whisper-large-v3`; `--model` accepts either spelling and ignores case.
With it, the scorer prints that model's CSV line for each dataset to stdout and
writes no files, so a single-model run cannot overwrite a full run's outputs.
Every model is scored independently against the reference, so the printed row is
identical to the one a full run writes.

Each `ref_rendering_<dataset>.csv` contains `rate`, `n`, and the same pair for
`spelling`, `initialism`, `number`, and `lexical`.

| column | meaning |
| --- | --- |
| `model` | model id from the results bucket |
| `rate` | `n_agreed / n`, over every eligible occurrence in the dataset. Empty when `n` is 0, which keeps "never had the chance" distinct from "never agreed" |
| `n` | eligible occurrences: reference spans carrying a candidate form where the model transcribed the same words in the same place and produced a form from the same family. An occurrence the model missed, misrecognized, or rendered outside the family is not counted either way |
| `<class>_rate` | the same ratio restricted to one class, `0.0` when that class has no eligible occurrence |
| `<class>_n` | eligible occurrences in that class; the four sum to `n` |

Rows are sorted by descending `rate`, models without any eligible occurrence
last. The classes are:

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
python ref_rendering/show_flagged_spans.py --preds_dir results \
  --dataset ami_test --class initialism --limit 20
```

Add `--html spans.html` for a browsable report.
