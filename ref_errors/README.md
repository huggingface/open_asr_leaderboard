# VoxPopuli reference error agreement

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

## Method

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

## Reproduce

```bash
pip install -r requirements/requirements_jobs.txt

# Sync the public results bucket and score every model.
python ref_errors/score_ref_errors.py --bucket hf-audio/asr_leaderboard_h200

# Or score an existing download.
python ref_errors/score_ref_errors.py --preds_dir results

# Score a single model, by Hub id, bucket id, or a unique substring of either.
python ref_errors/score_ref_errors.py --preds_dir results --model openai/whisper-large-v3
```

Manifest filenames cannot carry the `/` of a Hub id, so the bucket writes
`openai-whisper-large-v3`; `--model` accepts either spelling and ignores case.

With `--model`, the scorer prints that model's CSV line to stdout and writes no
files, so a single-model run cannot overwrite a full run's outputs. The row is
identical to the one a full run produces: the disagreement spans come from
diffing the two references, and each verdict depends only on that model's own
hypothesis.

The scorer needs prediction manifests for both `voxpopuli_test` and
`voxpopuli_cleaned_aa_test`. It reads only text manifests; no audio or inference is required.

## Outputs

`ref_error_agreement_voxpopuli.csv` contains one row per model:

| column | meaning |
| --- | --- |
| `model` | model id from the results bucket |
| `rate` | `n_ref / n_eligible` |
| `n_ref` | spans matching the original reference |
| `n_eligible` | spans matching either reference, namely (ref + consensus) above |
| `n_clips` | clips contributing eligible spans |

`edits_voxpopuli.jsonl` records every disagreement with its clip key, location,
original and corrected spans, distance, and per-model verdicts so results can be
audited directly.

To score another model, place its raw prediction manifest alongside the existing
manifests and rerun. Both bucket layouts are supported:

```
<preds-dir>/<model>/MODEL_<model>_DATASET_hf-audio-open-asr-leaderboard_<dataset>.jsonl
<preds-dir>/<dataset>/<model>.jsonl
```

## Limitations

- The rate covers disagreement spans, not the full dataset.
- Do not interpret small differences as a precise model ranking; spans from the
  same clip are related and denominators vary by model.
- It applies only where a trustworthy corrected reference exists. VoxPopuli is
  currently the only leaderboard dataset with the required pairing.
- A high rate is evidence of reference agreement, not proof of how that agreement
  arose.

## Attribution

Adapted from *Quantifying Benchmark Optimization in ASR Models*
([reference implementation](https://github.com/tlebryk/asr-benchmark-optimization),
Apache-2.0).
