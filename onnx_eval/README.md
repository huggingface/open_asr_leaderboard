# ONNX multilingual evaluation

This runner evaluates CPU-oriented ONNX ASR exports against the same 13 test
configurations used by the multilingual leaderboard. It supports two model
layouts:

- `onnx-asr`: NeMo Conformer TDT exports with `encoder-model.onnx`,
  `decoder_joint-model.onnx`, `config.json`, and `vocab.txt`.
- `sherpa-onnx`: NeMo transducer/TDT exports with separate encoder, decoder,
  joiner, and token files.

The Primeline workflow pins both model revisions and executes every model/config
pair in a separate GitHub-hosted CPU job. Model initialization, downloads, and
audio decoding are excluded from RTFx; ONNX preprocessing and decoding are
included. The aggregation job computes a duration-weighted RTFx across all 13
configs and validates that no config is missing before it can update
`scripts/data/multilingual.csv`.

Use **Actions → Primeline ONNX multilingual benchmark → Run workflow**. Set
`max_samples` to a small positive number for a two-job German smoke run, one job
per runtime. A publishable run uses `max_samples=0` and
`commit_results=true`; only that combination starts the 26-job matrix and
commits the two validated leaderboard rows.

The initial pinned variants are:

- FP32: `Buttermilk03/parakeet-primeline-onnx` at
  `2e2f9169c7030984cf0e9e35c0a68eadfac376e4`, loaded with `onnx-asr`.
- INT8: `flozen1981/parakeet-primeline-onnx` at
  `d548e25b9bfe559aa274f361892dc4ed5d64743a`, loaded with `sherpa-onnx`.
