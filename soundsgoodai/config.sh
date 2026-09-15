#!/bin/bash
# Shared evaluation settings for run_models.sh and submit_jobs.sh.

DEFAULT_DATASET_PATH=${DEFAULT_DATASET_PATH:-hf-audio/open-asr-leaderboard}

# Hub repository, export family, checkpoint filename, decoder, beam, batch size.
MODEL_CONFIGS=(
    "soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M zipformer model.pt                  transducer_modified_beam_search 10 256"
    "soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M zipformer model.pt                  ctc_greedy_search               1 256"
    "nvidia/parakeet-tdt-0.6b-v3                      parakeet  parakeet-tdt-0.6b-v3.nemo transducer_modified_beam_search 6 256"
    "nvidia/parakeet-tdt-0.6b-v2                      parakeet  parakeet-tdt-0.6b-v2.nemo transducer_modified_beam_search 6 256"
    "nvidia/parakeet-ctc-0.6b                         parakeet  parakeet-ctc-0.6b.nemo    ctc_greedy_search               1 256"
    "nvidia/parakeet-ctc-1.1b                         parakeet  parakeet-ctc-1.1b.nemo    ctc_greedy_search               1 256"
)

# Dataset label, split, optional repository (which uses its default config).
DATASET_CONFIGS=(
    "ami_cleaned test"
    "gigaspeech_cleaned test"
    "voxpopuli_cleaned_aa test"
    "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
    "librispeech test.clean"
    "librispeech test.other"
    "spgispeech test"
    "monsoon_en_in test VoiceArena/Monsoon_en_IN_test"
)

COMMON_ARGS=(
    --precision=fp16
    --min-audio-seconds=0.1
    --opt-audio-seconds=10.0
    --max-audio-seconds=40.0
    --optimization-level=5
    --warmup-steps=5
    --max-eval-samples=-1
    --device=0
)
