# Chilean Spanish ASR Evaluation

Streamlined evaluation framework for Automatic Speech Recognition (ASR) models on the Chilean Spanish dataset.

**Original Repository**: [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)
**Dataset**: [astroza/es-cl-asr-test-only](https://huggingface.co/datasets/astroza/es-cl-asr-test-only)

## Models Evaluated

This repository evaluates the following 7 ASR models:

1. **openai/whisper-large-v3** - OpenAI's Whisper Large V3 model
2. **openai/whisper-large-v3-turbo** - Faster variant of Whisper Large V3
3. **nvidia/canary-1b-v2** - NVIDIA's Canary multilingual ASR model
4. **nvidia/parakeet-tdt-0.6b-v3** - NVIDIA's lightweight Parakeet model
5. **microsoft/Phi-4-multimodal-instruct** - Microsoft's multimodal Phi-4 model
6. **mistralai/Voxtral-Mini-3B-2507** - Mistral AI's Voxtral ASR model
7. **elevenlabs/scribe_v1** - ElevenLabs' Scribe API-based model

## Metrics

We report two standard metrics:

- **WER (Word Error Rate)**: ⬇️ Lower is better
- **RTFx (Real-Time Factor)**: ⬆️ Higher is better (measures speed)

## Quick Start

### Prerequisites

1. Python 3.8+
2. CUDA-compatible GPU (recommended)
3. Hugging Face account (for accessing models/datasets)

### Installation

Choose the installation based on which models you want to evaluate:

#### Base Requirements (for Whisper models)
```bash
pip install -r requirements/requirements.txt
```

#### For NeMo models (Canary, Parakeet)
```bash
pip install -r requirements/requirements_nemo.txt
```

#### For Phi-4 model
```bash
pip install -r requirements/requirements_phi.txt
```

#### For API-based models (ElevenLabs)
```bash
pip install -r requirements/requirements-api.txt
```

#### All models
```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_nemo.txt
pip install -r requirements/requirements_phi.txt
pip install -r requirements/requirements-api.txt
```

**Note:** NeMo models require CUDA 12.6+ for RNN-T inference. Run `nvidia-smi` to check your CUDA version.

### Environment Setup

1. Connect your Hugging Face account:
```bash
huggingface-cli login
```

2. For API-based models, create a `.env` file in the root directory:
```bash
# For ElevenLabs Scribe
ELEVENLABS_API_KEY=your_api_key_here

# Optional: Hugging Face token for gated models
HF_TOKEN=your_hf_token_here
```

## Run Batch Evaluation

To evaluate all 7 models on the Chilean Spanish dataset:

```bash
./evaluate_chilean_asr.sh
```

To use a specific GPU:

```bash
DEVICE=0 ./evaluate_chilean_asr.sh
```

## Evaluate Individual Models

### Whisper Models (transformers)

```bash
python transformers/run_eval.py \
    --model_id "openai/whisper-large-v3" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 16 \
    --no-streaming
```

### NeMo Models (Canary, Parakeet)

```bash
python nemo_asr/run_eval.py \
    --model_id "nvidia/canary-1b-v2" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 32 \
    --no-streaming
```

### Phi-4 Model

```bash
python phi/run_eval.py \
    --model_id "microsoft/Phi-4-multimodal-instruct" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 4 \
    --no-streaming \
    --user_prompt "Transcribe el audio a texto en español."
```

### API-based Models (ElevenLabs)

```bash
python api/run_eval.py \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --model_name "elevenlabs/scribe_v1" \
    --max_workers 50
```

## Results

Results are saved as JSONL files in the `results/` directory with the following naming convention:

```
results/MODEL_<model_id>_DATASET_<dataset_path>_<dataset_name>_<split>.jsonl
```

Each result file contains:
- Audio filepath/ID
- Audio duration
- Transcription time
- Reference text (ground truth)
- Predicted text (model output)

Final metrics (WER and RTFx) are printed to console after each evaluation. 

## Repository Structure

```
.
├── evaluate_chilean_asr.sh    # Main batch evaluation script
├── transformers/               # Whisper and transformer-based models
│   └── run_eval.py
├── nemo_asr/                   # NVIDIA NeMo models (Canary, Parakeet)
│   └── run_eval.py
├── phi/                        # Microsoft Phi-4 model
│   └── run_eval.py
├── api/                        # API-based models (ElevenLabs)
│   └── run_eval.py
├── normalizer/                 # Text normalization utilities
│   ├── data_utils.py          # Dataset loading & Spanish text normalization
│   ├── eval_utils.py          # Metrics calculation (WER, RTFx)
│   └── normalizer.py          # Multilingual text normalizer
├── requirements/               # Framework-specific dependencies
│   ├── requirements.txt       # Base requirements
│   ├── requirements_nemo.txt  # NeMo models
│   ├── requirements_phi.txt   # Phi-4 model
│   └── requirements-api.txt   # API-based models
├── results/                    # Evaluation results (generated)
└── README.md                   # This file
```

## Text Normalization

The repository uses a **multilingual text normalizer** configured for Spanish:

- Preserves Spanish accents and special characters (á, é, í, ó, ú, ñ, etc.)
- Removes brackets, parentheses, and special symbols
- Normalizes whitespace
- Converts to lowercase

This is configured in `normalizer/data_utils.py` using `BasicMultilingualTextNormalizer(remove_diacritics=False)`.

## Notes

### Model-Specific Considerations

- **Phi-4**: Uses a custom prompt for Spanish transcription. Requires flash attention 2.
- **NeMo models**: Support batch inference and are optimized for speed.
- **Whisper models**: Support multiple languages out of the box.
- **Voxtral**: May require special handling; check if evaluation completes successfully.
- **ElevenLabs**: API-based, requires valid API key and credits.

### GPU Requirements

- **Minimum**: 16GB VRAM (for smaller models like Whisper, Parakeet)
- **Recommended**: 24GB+ VRAM (for Phi-4, Canary)
- **CPU**: Possible but very slow; not recommended

### Batch Sizes

Default batch sizes are optimized for a 24GB GPU. Adjust based on your hardware:

| Model | Default Batch Size | Min VRAM |
|-------|-------------------|----------|
| Whisper Large V3 | 16 | 16GB |
| Whisper Large V3 Turbo | 16 | 12GB |
| Canary 1B V2 | 32 | 20GB |
| Parakeet TDT 0.6B | 32 | 12GB |
| Phi-4 Multimodal | 4 | 24GB |
| Voxtral Mini 3B | 8 | 16GB |

## Troubleshooting

### Out of Memory (OOM) Errors
Reduce the batch size using the `--batch_size` parameter.

### Model Download Issues
Ensure you have sufficient disk space and a stable internet connection. Some models are large (10GB+).

### API Rate Limiting
For API-based models, reduce `--max_workers` or add delays between requests.

### CUDA Not Available
Ensure you have:
- NVIDIA GPU drivers installed
- CUDA toolkit installed
- PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Citation

If you use this evaluation framework or the Chilean Spanish dataset, please cite:

```bibtex
@misc{astroza2024chilean,
  title={Chilean Spanish ASR Test Dataset},
  author={Astroza},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/astroza/es-cl-asr-test-only}}
}

@misc{open-asr-leaderboard,
  title        = {Open Automatic Speech Recognition Leaderboard},
  author       = {Srivastav, Vaibhav and Majumdar, Somshubra and Koluguri, Nithin and Moumen, Adel and Gandhi, Sanchit and Hugging Face Team and Nvidia NeMo Team},
  year         = 2023,
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/open-asr-leaderboard/leaderboard}}
}
```

## License

This repository is derived from the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) and maintains compatibility with its evaluation protocols.
