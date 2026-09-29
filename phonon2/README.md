# Phonon-2 (FermionResearch/Phonon-2)

Five-value (sub-2-bit) Parakeet-TDT-0.6B encoder with 6-bit dense tables, shipped as a 177 MB container
(164 MB download). **Submitted path — NeMo:** `nemo_export.py` expands the container exactly to a dense `.nemo` of the base
architecture; `../nemo_asr/run_eval.py` then runs it as the board runs every stock Parakeet row (greedy_batch TDT =
label-looping decoder + CUDA graphs, bf16, batch 128, timestamps off). The WER is that of the shipped bytes; the RTFx is
NeMo's decoder on the dense expansion. The packed sub-2-bit kernels exist for Apple MLX only and are not submitted.
**Cross-check path — Transformers:** `run_eval.py` / `run_phonon2.sh`, the same exact expansion through the stock
`ParakeetForTDT` graph (bf16, batch 64); WER reproduction on a plain Transformers install, RTFx not submitted.

```bash
# from the repository root, after `pip install -r requirements/requirements_nemo.txt` (+ zstd)
bash phonon2/run_phonon2_nemo.sh            # SUBMITTED: export to .nemo, then all English short-form sets
bash phonon2/run_phonon2.sh                 # cross-check: Transformers path
MODEL_ID=/path/to/model.fermion bash phonon2/run_phonon2_nemo.sh   # a local container
```

Same decoding hyper-parameters on every dataset (greedy, no penalties). `run_eval.py` uses
`normalizer/data_utils.py` for loading/normalisation/manifests and saves raw predictions.

| Metadata | |
|---|---|
| License | cc-by-4.0 (inherited from nvidia/parakeet-tdt-0.6b-v3) |
| Size (B) | 0.6 (609 M parameters; encoder stored at < 2 bits/weight) |
| # Languages | 1 (English) |
| Encoder | FastConformer, about 2.1 bits per stored weight |
| Decoder | TDT / RNN-T |
| Training data disclosure | model card, "Training data" |
