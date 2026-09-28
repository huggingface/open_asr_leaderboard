# Expected leaderboard row for Phonon-2 (our run; the maintainers' H200 run is authoritative)
model | avg (7 sets) | RTFx | License | Size (B) | # Languages | Encoder | Decoder | AMI | E22 chunked | GigaSpeech | LS clean | LS other | SPGISpeech | VoxPopuli
FermionResearch/Phonon-2 | 5.15 (NeMo path, their scorer, A100-80; our meter 5.21) | 3,501 (NeMo greedy_batch TDT, bf16, batch 128, A100-80 — their H200 run replaces it) | cc-by-4.0 | 0.6 | 1 | FastConformer (low-bit) | TDT | 9.28 | 6.71 | 8.33 | 1.72 | 3.90 | 3.68 | 2.44
Voice Arena Monsoon (the board's 8th column): not measured by us (single-config repo; the kit's empty-config invocation is untested on our side); the maintainers' run supplies it, so the board's 8-column avg will differ from 5.21.
Placement among the board's ≤1B rows at commit d2c5b384: 11th of 27 on the seven shared columns, the smallest download in that set (164 MB vs 648 MB for the next).
