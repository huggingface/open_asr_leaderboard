# Reference rendering agreement

Models have varying degrees of exposure to the public benchmarks. Increased exposure can lead models to produce better transcripts on that benchmark, but may limit their generalizability. We cannot necessarily tell if a drop in WER comes from general competence or benchmark optimization.
One signal that a model has been heavily exposed to the benchmark's distribution is whether it reproduces the benchmark's verbatim formatting when multiple variants of the same span are available. These cases, when the same audio affords multiple textual renderings (e.g. "honour" vs "honor", "1" vs "one", "LCD" vs "L. C. D." etc.), are known as orthographic switches.

This directory adds per dataset scores for how often a model produces exact matches in these cases.

## Procedure
We use the normalizer as a starting point to flag spans containing an orthographic variant:

1. **Locate the spans.** Normalize the raw reference. We also align the
   transcript mapping by normalizing individual tokens, to determine which raw
   token mapped to which normalized token.
2. **A span whose raw and normalized forms differ is a flagged span.** This
   includes casing and punctuation.
3. **Label each flagged span** by the rewrite that produced it (table below).
4. **Filter for valid switch cases**, when variants are acoustically identical.
5. **Score each model at each remaining flagged span.** Per model, drop cases
   where the normalized span is absent from the model's normalized output. Then
   compare the raw (unnormalized) reference span with the raw model prediction.
   If the raw spans exactly match, increment the agreement count.

```
agreement rate = (flagged spans agreed) / (flagged spans the model was eligible at)
```

Note: denominators differ between models, because cases are dropped per model
when the normalized spans diverged.

## Classes

Every flagged span gets one label. First match wins, so the order below is part of
the definition. The reported rate is taken over the four scored classes:
per-token choices with no convention behind them. Casing and punctuation are
transcript-wide house style rather than per-token choices, so they are counted per
class but kept out of the rate.

| class         | rewrite                        | example (raw → normalized)       | in the rate  |
| ------------- | ------------------------------ | -------------------------------- | ------------ |
| `contraction` | apostrophe form expanded       | `don't` → `do not`               | dropped      |
| `disfluency`  | filler deleted                 | `Uh` → ∅                         | dropped      |
| `case`        | capitalization only            | `So` → `so`                      | counted only |
| `punct`       | punctuation or spacing only    | `yeah,` → `yeah`                 | counted only |
| `spelling`    | en-GB/en-US map entry          | `colour` → `color`               | **scored**   |
| `abbrev`      | abbreviated honorific expanded | `mr` → `mister`                  | **scored**   |
| `acronym`     | pointed initialism flattened   | `U.S.` → `u s`                   | **scored**   |
| `number`      | digits against number words    | `forty` → `40`, `1 000` → `1000` | **scored**   |
| `other`       | any other rewrite              | `gonna` → `going to`             | counted only |

One number is reported per dataset. `dropped` spans leave the data entirely;
`counted only` ones keep a per-class count in the CSV for inspection, and never
enter the reported rate.

## How many flagged spans there are

Retained spans per dataset, reference side, before the per-model eligibility drop.
`pool` is the sum of the four scored classes that follow it. The pool is a small
fraction of what the normalizer rewrites, and its composition differs by corpus:
VoxPopuli is mostly spelling and honorifics, spgispeech is the only set where
acronyms carry weight, and GigaSpeech is almost entirely numbers.

| dataset                     | clips  | pool  | spelling | abbrev | acronym | number | case   | punct  |
| --------------------------- | ------ | ----- | -------- | ------ | ------- | ------ | ------ | ------ |
| `spgispeech_test`           | 39,341 | 929   | 71       | 43     | 154     | 661    | 56,644 | 80,161 |
| `gigaspeech_test`           | 19,856 | 3,389 | 103      | 28     | 6       | 3,252  | 0      | 38,382 |
| `gigaspeech_cleaned_test`   | 18,757 | 3,372 | 101      | 28     | 6       | 3,237  | 0      | 37,273 |
| `ami_test`                  | 11,626 | 537   | 209      | 6      | 1       | 321    | 6,626  | 14,681 |
| `ami_cleaned_test`          | 7,715  | 512   | 200      | 4      | 0       | 308    | 5,719  | 11,105 |
| `voxpopuli_test`            | 1,842  | 451   | 174      | 60     | 0       | 217    | 0      | 2,065  |
| `librispeech_test.other`    | 2,939  | 198   | 57       | 0      | 0       | 141    | 0      | 12     |
| `librispeech_test.clean`    | 2,620  | 175   | 73       | 1      | 0       | 101    | 0      | 2      |
| `earnings22_test`           | 2,737  | 146   | 9        | 8      | 3       | 126    | 3,101  | 7,230  |
| `voxpopuli_cleaned_aa_test` | 628    | 77    | 30       | 24     | 0       | 23     | 1,609  | 1,558  |

Counts are from one manifest (`openai-whisper-large-v3`); manifests occasionally
differ in row count by under 0.1%, so treat them as approximate.

`voxpopuli_test`, both GigaSpeech sets and both LibriSpeech sets have no `case`
spans, because their published references are uniformly lowercase or uppercase.
LibriSpeech has almost no `punct` spans either: its references carry no
punctuation to begin with. A corpus that pre-normalized its transcripts offers
less to measure, which is a property of the corpus rather than of the models.

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

## Limitations

- **English only.** The construction needs the English normalizer's rewrite
  behaviour, so it applies to the English short-form sets.
- **Classification order.** `classify` returns the first matching label, and
  `punct` is tested before `acronym`. A pointed initialism whose punctuation
  removal alone accounts for the rewrite is therefore labelled `punct` and left out
  of the rate. This covers the spaced form: `L. C. D.` becomes three `punct` spans,
  and only glued forms such as `U.S.` reach `acronym`. Affected spans, nearly all
  single-letter initials (`T.` → `t`): 815 on `ami_test`, 791 on
  `ami_cleaned_test`, 219 on `gigaspeech_test`, 10 on `spgispeech_test`, none
  elsewhere. Treat the `acronym` column as a lower bound on acronym volume.
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
- **Numbers dominate the pool on some corpora.** After pruning, the `number` class
  still contains years and other multi-digit tokens with more than one natural
  spoken form, so part of it reflects which form was heard rather than how it was
  written. It is also 96% of the pool on GigaSpeech and 86% on earnings22, against
  48% on VoxPopuli, so the same rate does not weigh the same evidence on every
  dataset. The per-class columns exist so the number contribution can be inspected
  or set aside.

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

`ref_rendering_<dataset>.csv` — one row per model, sorted by descending rate:

| column                      | meaning                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| `model`                     | model id as it appears in the results bucket                     |
| `rate`, `n`                 | agreement rate over the scored classes, and its denominator      |
| `lo`, `hi`                  | 95% Wilson interval on `rate`                                    |
| `<class>_rate`, `<class>_n` | per-class breakdown, for inspection; not reported                |

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
