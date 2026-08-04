# Reference error agreement rate on VoxPopuli

VoxPopuli's English references derive from parliamentary records rather than from
the audio, so some spans they contain were never spoken and some spoken words
they omit. The [`ArtificialAnalysis/VoxPopuli-Cleaned-AA`](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA)
subset supplies a human-corrected reference for 628 clips of `voxpopuli_test`,
and the leaderboard already reports WER against it (`Voxpopuli-Cleaned-AA WER`).

Diffing the official reference against the correction, under the same text
normalization the leaderboard's WER uses, locates 564 spans on which the two
disagree. For each span a model's output either matches the correction or
reproduces the official reference. The **reference error agreement rate** is the
share of spans where a model reproduces the official reference:

```
agreement rate = (spans matching the official reference) / (spans the model was scored on)
```

A model transcribing the audio produces the correction's version. A high rate
therefore means a model reproduces reference spans that human correction shows
were never spoken. It is a rate over reference-audio disagreements, not a WER,
and is not comparable to one.

## Reproduce

```bash
pip install -r requirements/requirements_jobs.txt

# Syncs the public results bucket, then scores every model in it.
python ref_errors/score_ref_errors.py --bucket hf-audio/asr_leaderboard_h200
```

Nothing is inferred and no audio is read: both reference sets and all
hypotheses come from prediction manifests already published in the bucket. The
`text` field of the `voxpopuli_test` manifests carries the official reference,
the `text` field of the `voxpopuli_cleaned_aa_test` manifests carries the
correction, and the `pred_text` field of the latter carries the hypotheses. To
re-score an already-downloaded copy:

```bash
python ref_errors/score_ref_errors.py --preds-dir results
```

## Outputs

`ref_error_agreement_voxpopuli.csv` — one row per model:

| column | meaning |
| --- | --- |
| `model` | model id as it appears in the results bucket |
| `rate` | agreement rate, `n_ref / n_eligible` |
| `lo`, `hi` | 95% Wilson interval on `rate` |
| `n_ref` | spans where the output matched the official reference |
| `n_eligible` | spans the model was scored on |

`edits_voxpopuli.jsonl` — one row per disagreement: clip key, `kind`
(`delete` where the official reference carries a span the correction removes or
replaces, `insert` where the correction adds a span the official reference
omits), `position` (`start` / `middle` / `end` of the reference),
`official_span`, `corrected_span`, the reference token indices, and the
character error rate between the two spans.

## Scoring a new model

Place the model's `voxpopuli_cleaned_aa_test` prediction manifest alongside the
others and re-run. Either layout is read:

```
<preds-dir>/<model>/MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_cleaned_aa_test.jsonl
<preds-dir>/voxpopuli_cleaned_aa_test/<model>.jsonl
```

The manifest must be the one `normalizer.eval_utils.write_manifest` produces, so
that `text` carries the corrected reference and `pred_text` the hypothesis. A
`voxpopuli_test` manifest for at least one model must also be present, since the
official reference is read from there; it need not be the same model.

## Interpretation notes

- **Denominators differ between models and must be reported.** A model is scored
  on a clip only if its output reproduces at least half the reference tokens
  (`min_ref_match`), so empty, truncated and off-language outputs are dropped
  rather than counted as agreement with the correction. Report `n_eligible`
  alongside `rate`.
- **Spans that survive normalization only.** A `delete` span is kept only if what
  the correction puts in its place is character-wise far from it
  (`min_consensus_cer = 0.30`), which removes spacing, accent and spelling
  differences.
- **Insertions are boundary-only.** Which side of a matched token an inserted
  word belongs to is an alignment choice rather than a fact about the audio, so
  only the two reference boundaries are used. Deletions are counted throughout
  (`include_middle = True`).
- **WER is not a substitute.** WER against the official reference rewards
  reproducing these spans and WER against the correction penalises it, but in
  both cases the effect is a few tenths of a point diluted across every other
  token in the clip. Restricting to the disagreements is what makes the
  quantity readable.
- **Confidence intervals are wide.** 564 spans over 628 clips; Wilson intervals
  on adjacent models overlap substantially. Read the interval, not the ordering.
- **English only, one benchmark.** The construction needs a human-corrected
  reference for a benchmark whose official reference is known to diverge from the
  audio. VoxPopuli is the only such case currently on the leaderboard.

## Parameters

Passed unchanged:
`min_run_len = 1`, `include_middle = True`, `min_ref_match = 0.5`,
`min_consensus_cer = 0.30` (`ref_error_utils.DEFAULTS`).

## Attribution

Adapted from the reference implementation accompanying "Quantifying Benchmark
Optimization in ASR Models"
([tlebryk/asr-benchmark-optimization](https://github.com/tlebryk/asr-benchmark-optimization),
Apache-2.0), modules `align.py` and `refdis.py`. The alignment helpers are
carried over verbatim; the edit finder is specialised to a single human-corrected
reference in place of that implementation's model panel.
