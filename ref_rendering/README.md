# Reference rendering agreement

Models have varying degrees of exposure to the public benchmarks. Increased exposure can lead models to produce better transcripts on that benchmark, but may limit their generalizability. We cannot necessarily tell if a drop in WER comes from general competence or benchmark optimization.
One signal that a model has been heavily exposed to the benchmark's distribution is whether it reproduces the benchmark's verbatim formatting when multiple variants of the same span are available. These cases, when the same audio affords multiple textual renderings (e.g. "honour" vs "honor", "1" vs "one", "LCD" vs "L. C. D." etc.), are known as orthographic switches.

This directory adds per dataset scores for how often it produces exact matches in these cases. 

## Procedure
We use the normalizer as a starting point to flag orthographic switches:

1. **Locate the switches.** Normalize the raw reference, then align its raw
   whitespace tokens against the normalized words: each raw token is normalized
   on its own and that sequence is aligned against the full-string
   normalization. This gives a raw-span to normalized-span map. Tokens a single
   rule rewrites jointly (`1 000` to `1000`) are merged, since they have no
   per-token alignment.
2. **A span whose raw and normalized forms differ is a flagged span.** Nothing else
   is consulted at this step, so casing and punctuation come out of the same pass
   as spelling: `So` to `so` and `yeah,` to `yeah` are rewrites like any other.
3. **Label each flagged span** by the rewrite that produced it (table below). The
   en-GB/en-US map is used only here, as a labeller — the spans were already found
   in step 2.
4. **Drop the flagged spans where the reference had no real choice**, by the two
   tests in *Which flagged spans count*.
5. **Score each model at each remaining flagged span.** Align the normalized
   reference against the model's normalized output. It is **eligible** only if its
   normalized span falls inside a block that alignment marks equal — the model
   produced those words in that position, so only the rendering is in question.
   The model **agrees** if the raw output aligned to that span equals the
   reference's raw span exactly.

```
agreement rate = (flagged spans agreed) / (flagged spans the model was eligible at)
```

Agreement is exact string equality at the aligned position, not anywhere in the
clip. Spans where the model got the words wrong are dropped rather than counted as
disagreement, so denominators differ between models and must be reported.

## Classes

First match wins, so the order below is part of the definition. The last two
columns are retained span counts, to show what each pool is actually made of.

| class         | rewrite                        | example (raw → normalized)       | V1  | V2  | vox   | ami    |
| ------------- | ------------------------------ | -------------------------------- | --- | --- | ----- | ------ |
| `contraction` | apostrophe form expanded       | `don't` → `do not`               | –   | –   | –     | –      |
| `disfluency`  | filler deleted                 | `Uh` → ∅                         | –   | –   | –     | –      |
| `case`        | capitalization only            | `So` → `so`                      | ✓   | –   | 0     | 6,626  |
| `punct`       | punctuation or spacing only    | `yeah,` → `yeah`                 | ✓   | –   | 2,065 | 14,704 |
| `spelling`    | en-GB/en-US map entry          | `colour` → `color`               | ✓   | ✓   | 174   | 209    |
| `abbrev`      | abbreviated honorific expanded | `mr` → `mister`                  | ✓   | ✓   | 60    | 6      |
| `acronym`     | pointed initialism flattened   | `U.S.` → `u s`                   | ✓   | ✓   | 0     | 1      |
| `number`      | digits against number words    | `forty` → `40`, `1 000` → `1000` | ✓   | ✓   | 217   | 321    |
| `other`       | any other rewrite              | `gonna` → `going to`             | ✓   | –   | 2     | 787    |

`voxpopuli_test` has no `case` spans because its published references are already
lowercase, which is why V1 and V2 diverge most sharply there.

## Which flagged spans count

Two tests decide it.

**Audibility.** A flagged span counts only if the audio does not determine the
rendering. Whether a contraction was spoken contracted, and whether a filler was
uttered at all, are facts about the audio, so `contraction` and `disfluency` are
excluded outright.

**Canonicity.** A flagged span counts only if the reference's rendering was a choice
rather than the only correct form. Two rules follow from it.

*Spelling sense-blocklist.* The map has 1,743 entries, of which 84 fire on
`voxpopuli_test` and 71 on `ami_test`. Some are pairs whose American form is also
an ordinary English word with a different sense; there the reference's spelling is
fixed by meaning. Blocked, keyed on the normalized form (plural forms likewise):

| pair                       | why blocked                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| `cheque` / `check`         | the American form is the ordinary verb, and the idiom "checks and balances" |
| `programme` / `program`    | the software sense is spelled `program` in en-GB as well                    |
| `connexion` / `connection` | `connexion` is archaic rather than a live variant                           |
| `tonne` / `ton`            | "tons of" is an intensifier, not a unit of mass                             |
| `practise` / `practice`    | en-GB spells the noun `practice`; only the verb differs                     |
| `draught` / `draft`        | a draft document is a different word, not a variant spelling                |
| `storey` / `story`         | the narrative sense is spelled `story` in en-GB as well                     |
| `metre` / `meter`          | the measuring-device sense is spelled `meter` in en-GB as well              |
| `philtre` / `filter`       | a philtre is a love potion; the pair is miscategorized                      |
| `biassed` / `biased`       | `biassed` is vanishingly rare in either dialect                             |
| `tyre` / `tire`            | the verb "to tire" is spelled the same in en-GB                             |

*Number pruning.* `oh` and `o` for zero are dropped: how a digit was spoken is
audible, not a way of writing it. Ordinals up to `10th` and bare integers below
`11` are dropped: at those magnitudes spelling the number out is the near-uniform
convention of edited prose, so the reference is following a rule rather than
choosing.

## V1 and V2

**V1** pools every retained class. It is dominated by `case` and `punct`, which
are transcript-wide conventions — whether to capitalize sentence-initially,
whether to punctuate at all — so V1 largely measures whether a model's output
follows the benchmark's house style.

**V2** pools `spelling`, `abbrev`, `acronym` and `number`: per-token choices with
no house-style rule behind them, where two renderings of the same audio are both
correct English.

They are not the same measurement: across the 73 models scored on the two
datasets inspected during development, V1 and V2 correlate weakly (Pearson 0.10
and 0.28; Spearman 0.40 and 0.26), and models near the top of one are routinely
mid-table on the other.

## Interpretation

- **A high rate means the output's formatting tracks this benchmark's published
  transcripts.** It does not identify why, and there are innocent reasons. A
  product aiming at verbatim transcription, or one whose output style happens to
  coincide with a benchmark's conventions, can rate high on that benchmark
  legitimately.
- **The informative read is one model across datasets.** The benchmarks differ in
  register and in transcription convention, while a model's output style is
  largely fixed, so a model that rates high everywhere is telling a different
  story from one that rates high on a single benchmark's conventions.
- **Denominators differ between models and must be reported.** Eligibility
  requires reproducing the span's words, so a weaker model is scored on fewer
  spans. Report `v2_n` alongside `v2_rate`, and read the Wilson interval rather
  than the ordering: adjacent models overlap.
- **This is a rate over formatting choices, not a WER**, and not comparable to
  one.
- **English only.** The construction needs the English normalizer's rewrite
  behaviour, so it applies to the English short-form sets.

## Limitations

- **Classification order.** `classify` returns the first matching label, and
  `punct` is tested before `acronym`. A pointed initialism whose punctuation
  removal alone accounts for the rewrite is therefore labelled `punct` and pooled
  into V1, not V2. This covers the spaced form: `L. C. D.` becomes three `punct`
  spans, and only glued forms such as `U.S.` reach `acronym`. On one
  conversational dataset it affects 813 spans, nearly all single-letter initials
  (`T.` → `t`). The behaviour is left as it is because
  the frozen rates are defined over it; treat the `acronym` column as a lower
  bound on acronym volume.
- **One-armed detection.** Only spans where the reference departs from the
  normalizer's canonical form are visible. A reference already in canonical form
  (`honor`, `US`, `2019`) is never flagged, although the model faced the same
  choice there. Recovering that arm by inverting the normalizer's tables was
  implemented and audited by hand, up to 40 sampled spans per class and dataset:
  false-positive rates were 30–91% for spelling, 10–92% for acronyms and 58% for
  numbers, with only expanded honorifics clean (0%, n = 9). The inverse is not a
  function — an American spelling is usually also the only correct spelling for
  the sense used, a bare acronym pronounced as a word admits no pointed form, and
  a four-digit year has several natural spoken forms — so it is not shipped.
  Rates are therefore conditional on the choices the reference itself made.
- **Number-class residual.** After pruning, the class still contains years and
  other multi-digit tokens with more than one natural spoken form, so part of it
  reflects which form was heard rather than how it was written. The per-class
  columns exist so the number contribution can be inspected or set aside.
- **Non-pairs in the spelling map.** The map this repo ships contains a few
  entries that are not en-GB/en-US pairs (`ok` and `'kay` to `okay`, `etcetera`
  to `etc`). Each was arbitrated with the audibility test: `'kay` is clipped
  speech, audibly distinct from `okay`, so it is excluded via
  `BLOCKED_RAW_SPELLINGS`; `ok`/`OK` vs `okay` and `etc` vs `etcetera` are read
  identically, so they are kept as genuine free-variant spans. The kept entries
  are rare (a few dozen references across the two datasets inspected) and
  concentrate in conversational transcripts.

## Reproduce

```bash
pip install -r requirements/requirements_jobs.txt

# Syncs the public results bucket, then scores every English short-form set in it.
python ref_rendering/score_ref_rendering.py --bucket hf-audio/asr_leaderboard_h200

# Re-score an already-downloaded copy, one dataset.
python ref_rendering/score_ref_rendering.py --preds-dir results --datasets voxpopuli_test
```

Both the reference and the hypothesis of each row come from the same manifest, so
no cross-file join is needed and manifests keyed `sample_<i>` are usable.

## Outputs

`ref_rendering_<dataset>.csv` — one row per model, sorted by descending V2:

| column                      | meaning                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `model`                     | model id as it appears in the results bucket                  |
| `v1_rate`, `v1_n`           | agreement rate over all retained classes, and its denominator |
| `v2_rate`, `v2_n`           | agreement rate over `spelling`, `abbrev`, `acronym`, `number` |
| `v2_lo`, `v2_hi`            | 95% Wilson interval on `v2_rate`                              |
| `<class>_rate`, `<class>_n` | the same rate per class, as a diagnostic                      |

## Auditing the flagged spans

`show_flagged_spans.py` prints a sample with six words of context either side, the
reference's raw span, what the normalizer makes of it, and each model's raw
output aligned to it:

```bash
python ref_rendering/show_flagged_spans.py --preds-dir results --dataset voxpopuli_test \
    --models model-a,model-b --class spelling --limit 20 --seed 7

# Same thing as a page.
python ref_rendering/show_flagged_spans.py --preds-dir results --dataset ami_test \
    --limit 30 --seed 7 --html spans_ami.html
```

Models are joined on the reference text, so only clips every selected model
transcribed are shown. Sampling is seeded, so a cited span is reproducible.

## Scoring a new model

Place the model's prediction manifest alongside the others and re-run. Either
layout is read:

```
<preds-dir>/<model>/MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_<dataset>.jsonl
<preds-dir>/<dataset>/<model>.jsonl
```

The manifest must be the one `normalizer.eval_utils.write_manifest` produces, so
that `text` carries the reference and `pred_text` the hypothesis. Predictions
must not be normalized before being written, since the raw rendering is the
measurement.

## Attribution

Adapted from the reference implementation accompanying "Quantifying Benchmark
Optimization in ASR Models"
([tlebryk/asr-benchmark-optimization](https://github.com/tlebryk/asr-benchmark-optimization),
Apache-2.0). Extraction, classification and scoring are carried over; the
normalizer is this repo's own.
