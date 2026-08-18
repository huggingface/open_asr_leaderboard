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
  --datasets voxpopuli_test
```

Each `ref_rendering_<dataset>.csv` contains `rate`, `n`, and the same pair for
`spelling`, `initialism`, `number`, and `lexical`.

To inspect the raw examples:

```bash
python ref_rendering/show_flagged_spans.py --preds_dir results \
  --dataset ami_test --class initialism --limit 20
```

Add `--html spans.html` for a browsable report.
