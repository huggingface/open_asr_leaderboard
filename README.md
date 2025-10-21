# Chilean Spanish ASR Evaluation

> **Specialized adaptation of the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) for evaluating Automatic Speech Recognition (ASR) models on Chilean Spanish.**

## About This Repository

This repository is a **streamlined, task-specific version** of the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) evaluation framework, specifically adapted for benchmarking ASR models on the **Chilean Spanish dialect**.

### What is the Open ASR Leaderboard?

The [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) is a comprehensive benchmarking framework developed by Hugging Face, NVIDIA NeMo, and the community to evaluate ASR models across multiple English datasets (LibriSpeech, AMI, VoxPopuli, Earnings-22, GigaSpeech, SPGISpeech, TED-LIUM). It supports various ASR frameworks including Transformers, NeMo, SpeechBrain, and more, providing standardized WER and RTFx metrics.

### How This Repository Differs

This Chilean Spanish adaptation makes the following key modifications to focus exclusively on Chilean Spanish ASR evaluation:

| Aspect | Original Open ASR Leaderboard | This Repository |
|--------|-------------------------------|-----------------|
| **Target Language** | English (primarily) | Chilean Spanish |
| **Dataset** | 7 English datasets (LibriSpeech, AMI, etc.) | Single dataset: [`astroza/es-cl-asr-test-only`](https://huggingface.co/datasets/astroza/es-cl-asr-test-only) |
| **Text Normalization** | English text normalizer | **Multilingual normalizer** preserving Spanish accents (á, é, í, ó, ú, ñ) |
| **Model Focus** | Broad coverage (~50+ models) | **7 selected models** optimized for multilingual/Spanish ASR |
| **Frameworks** | 10+ frameworks (Transformers, NeMo, SpeechBrain, CTranslate2, etc.) | **4 frameworks**: Transformers, NeMo, Phi, API |
| **Batch Script** | Individual scripts per model type | **Single unified script** (`evaluate_chilean_asr.sh`) for all models |
| **Repository Size** | ~15,000+ lines of code | **~3,000 lines** (streamlined) |

### Key Modifications

1. **Text Normalization for Spanish**
   - Switched from `EnglishTextNormalizer()` to `BasicMultilingualTextNormalizer(remove_diacritics=False)`
   - Preserves critical Spanish characters: `á, é, í, ó, ú, ñ, ü, ¿, ¡`
   - Modified in `normalizer/data_utils.py`

2. **Removed Unused Frameworks**
   - Deleted: SpeechBrain, Moonshine, Kyutai, Granite, CTranslate2, TensorRT-LLM, LiteASR
   - Kept only frameworks needed for the 7 selected models

3. **Simplified Dataset Configuration**
   - Single dataset configuration: `astroza/es-cl-asr-test-only`
   - Removed multi-dataset evaluation logic

4. **Unified Batch Evaluation**
   - Created `evaluate_chilean_asr.sh` to run all 7 models sequentially
   - Pre-configured with optimal batch sizes and parameters for each model

5. **Chilean Spanish-Specific Documentation**
   - Updated README with Chilean Spanish context
   - Added Spanish-specific troubleshooting and considerations

---

## Models Evaluated

This repository evaluates **7 state-of-the-art ASR models** selected for their multilingual or Spanish language support:

| Model | Type | Framework | Parameters | Notes |
|-------|------|-----------|------------|-------|
| **openai/whisper-large-v3** | Multilingual | Transformers | 1.5B | OpenAI's flagship ASR model |
| **openai/whisper-large-v3-turbo** | Multilingual | Transformers | 809M | Faster Whisper variant |
| **nvidia/canary-1b-v2** | Multilingual | NeMo | 1B | NVIDIA's multilingual ASR |
| **nvidia/parakeet-tdt-0.6b-v3** | Multilingual | NeMo | 0.6B | Lightweight, fast inference |
| **microsoft/Phi-4-multimodal-instruct** | Multimodal | Phi | 14B | Microsoft's multimodal LLM with audio |
| **mistralai/Voxtral-Mini-3B-2507** | Speech-to-text | Transformers | 3B | Mistral's ASR model |
| **elevenlabs/scribe_v1** | API-based | API | N/A | ElevenLabs' commercial ASR API |

## Dataset

**Dataset**: [`astroza/es-cl-asr-test-only`](https://huggingface.co/datasets/astroza/es-cl-asr-test-only)
**Language**: Spanish (Chilean variant)
**Split**: `test`
**Domain**: Chilean Spanish speech samples

## Metrics

Following the Open ASR Leaderboard standard, we report:

- **WER (Word Error Rate)**: ⬇️ Lower is better - Measures transcription accuracy
- **RTFx (Real-Time Factor)**: ⬆️ Higher is better - Measures inference speed (audio_duration / transcription_time)

---

## Quick Start

### Prerequisites

1. Python 3.8+
2. CUDA-compatible GPU (recommended: 24GB+ VRAM)
3. Hugging Face account
4. Sufficient disk space (~50GB for all models)

### Installation

#### Option 1: Install All Dependencies

```bash
# Clone the repository
git clone https://github.com/aastroza/open_asr_leaderboard_cl.git
cd open_asr_leaderboard_cl

# Install all framework dependencies
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_nemo.txt
pip install -r requirements/requirements_phi.txt
pip install -r requirements/requirements-api.txt
```

#### Option 2: Install Selectively

Install only what you need for specific models:

```bash
# For Whisper and Voxtral (transformers framework)
pip install -r requirements/requirements.txt

# For NeMo models (Canary, Parakeet)
pip install -r requirements/requirements_nemo.txt

# For Phi-4 Multimodal
pip install -r requirements/requirements_phi.txt

# For ElevenLabs Scribe (API-based)
pip install -r requirements/requirements-api.txt
```

**Important**: NeMo models require CUDA 12.6+ for optimal performance. Check with:
```bash
nvidia-smi  # Should show "CUDA Version: 12.6" or higher
```

### Environment Setup

1. **Authenticate with Hugging Face** (required for downloading models/datasets):
```bash
huggingface-cli login
```

2. **Configure API keys** (optional, only for ElevenLabs):

Create a `.env` file in the repository root:
```bash
# For ElevenLabs Scribe V1
ELEVENLABS_API_KEY=your_api_key_here

# Optional: For gated models
HF_TOKEN=your_hf_token_here
```

---

## Usage

### Batch Evaluation (All 7 Models)

Evaluate all models on the Chilean Spanish dataset with a single command:

```bash
./evaluate_chilean_asr.sh
```

**GPU Selection:**
```bash
# Use GPU 0 (default)
DEVICE=0 ./evaluate_chilean_asr.sh

# Use GPU 1
DEVICE=1 ./evaluate_chilean_asr.sh
```

The script will:
1. Evaluate each model sequentially
2. Save results to `results/*.jsonl`
3. Print WER and RTFx for each model
4. Handle errors gracefully (continues on failure)

**Estimated Runtime** (on NVIDIA A100 80GB):
- All 7 models: ~2-4 hours (depends on dataset size)

### Individual Model Evaluation

#### Whisper Models

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

**For Whisper Large V3 Turbo:**
```bash
python transformers/run_eval.py \
    --model_id "openai/whisper-large-v3-turbo" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 16 \
    --no-streaming
```

#### NeMo Models (Canary, Parakeet)

**NVIDIA Canary 1B V2:**
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

**NVIDIA Parakeet TDT 0.6B:**
```bash
python nemo_asr/run_eval.py \
    --model_id "nvidia/parakeet-tdt-0.6b-v3" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 32 \
    --no-streaming
```

#### Microsoft Phi-4 Multimodal

```bash
python phi/run_eval.py \
    --model_id "microsoft/Phi-4-multimodal-instruct" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 4 \
    --no-streaming \
    --max_new_tokens 128 \
    --warmup_steps 2 \
    --user_prompt "Transcribe el audio a texto en español."
```

**Note**: The Spanish prompt is critical for this model.

#### Mistral Voxtral

```bash
python transformers/run_eval.py \
    --model_id "mistralai/Voxtral-Mini-3B-2507" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --device 0 \
    --batch_size 8 \
    --no-streaming \
    --max_new_tokens 128
```

#### ElevenLabs Scribe (API)

```bash
python api/run_eval.py \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --split "test" \
    --model_name "elevenlabs/scribe_v1" \
    --max_workers 50
```

**Note**: Requires `ELEVENLABS_API_KEY` in `.env`. API usage will incur costs.

---

## Results

### Output Format

Results are saved as JSONL (JSON Lines) files in the `results/` directory:

```
results/MODEL_openai-whisper-large-v3_DATASET_astroza_es-cl-asr-test-only_test.jsonl
```

Each line contains:
```json
{
  "audio_filepath": "sample_001",
  "duration": 12.5,
  "time": 0.8,
  "text": "hola cómo estás",
  "pred_text": "hola como estas"
}
```

### Interpreting Results

**Console Output Example:**
```
WER: 15.3 %
RTFx: 42.5
```

- **WER = 15.3%**: The model made errors on 15.3% of words
- **RTFx = 42.5**: The model transcribes 42.5× faster than real-time (12 seconds of audio takes ~0.28 seconds)

**What's a Good Score?**
- **WER**: <10% = Excellent, 10-20% = Good, 20-30% = Fair, >30% = Poor
- **RTFx**: >10 = Fast enough for real-time, >50 = Very fast, <1 = Slower than real-time

---

## Repository Structure

```
open_asr_leaderboard_cl/
├── evaluate_chilean_asr.sh       # Batch evaluation script for all models
│
├── transformers/                  # Hugging Face Transformers framework
│   ├── run_eval.py               # Evaluation script for Whisper, Voxtral
│   └── run_*.sh                  # Model-specific batch scripts
│
├── nemo_asr/                      # NVIDIA NeMo framework
│   ├── run_eval.py               # Evaluation script for Canary, Parakeet
│   └── run_*.sh                  # Model-specific batch scripts
│
├── phi/                           # Microsoft Phi framework
│   ├── run_eval.py               # Evaluation script for Phi-4 Multimodal
│   └── run_phi4_multimodal.sh    # Batch script
│
├── api/                           # API-based models
│   ├── run_eval.py               # Evaluation script for ElevenLabs
│   └── run_api.sh                # Batch script
│
├── normalizer/                    # Text normalization utilities
│   ├── data_utils.py             # Dataset loading & Spanish text normalization
│   ├── eval_utils.py             # Metrics calculation (WER, RTFx)
│   ├── normalizer.py             # Multilingual text normalizer
│   └── english_abbreviations.py  # English abbreviations (legacy)
│
├── requirements/                  # Framework-specific dependencies
│   ├── requirements.txt          # Base: torch, transformers, datasets
│   ├── requirements_nemo.txt     # NeMo ASR (Canary, Parakeet)
│   ├── requirements_phi.txt      # Phi-4 (flash-attn, peft)
│   └── requirements-api.txt      # API clients (ElevenLabs, OpenAI)
│
├── results/                       # Evaluation results (auto-generated)
│   └── *.jsonl                   # Per-model result files
│
└── README.md                      # This file
```

---

## Text Normalization for Spanish

### Why This Matters

Spanish uses **diacritical marks** (accents) that change word meaning:
- `esta` (this) vs. `está` (is)
- `si` (if) vs. `sí` (yes)
- `el` (the) vs. `él` (he)

The original Open ASR Leaderboard uses an **English text normalizer** that removes all diacritics, which is inappropriate for Spanish evaluation.

### Our Approach

This repository uses a **multilingual normalizer** configured to preserve Spanish characters:

```python
# In normalizer/data_utils.py
normalizer = BasicMultilingualTextNormalizer(remove_diacritics=False)
```

**What it does:**
- ✅ Preserves: `á, é, í, ó, ú, ñ, ü, ¿, ¡`
- ✅ Removes: Brackets `[...]`, parentheses `(...)`, special symbols
- ✅ Normalizes: Whitespace, capitalization (converts to lowercase)
- ❌ Does NOT remove: Accents or Spanish-specific characters

**Example:**
```python
Input:  "¿Cómo estás? [ruido] (suspiro)"
Output: "cómo estás"
```

---

## GPU Requirements & Batch Sizes

### Recommended Hardware

| Model | Min VRAM | Recommended VRAM | Default Batch Size | CPU Possible? |
|-------|----------|------------------|-------------------|---------------|
| Whisper Large V3 | 16GB | 24GB | 16 | ⚠️ Very slow |
| Whisper V3 Turbo | 12GB | 16GB | 16 | ⚠️ Very slow |
| Canary 1B V2 | 20GB | 24GB | 32 | ❌ No |
| Parakeet TDT 0.6B | 12GB | 16GB | 32 | ⚠️ Very slow |
| Phi-4 Multimodal | 24GB | 40GB | 4 | ❌ No |
| Voxtral Mini 3B | 16GB | 24GB | 8 | ⚠️ Very slow |
| ElevenLabs Scribe | N/A (API) | N/A | N/A | ✅ Yes |

### Adjusting Batch Size

If you encounter **Out of Memory (OOM)** errors, reduce the batch size:

```bash
# Example: Reduce Whisper batch size from 16 to 8
python transformers/run_eval.py \
    --model_id "openai/whisper-large-v3" \
    --batch_size 8 \  # Reduced from default 16
    ...
```

### Multi-GPU Evaluation

To run evaluations across multiple GPUs in parallel:

```bash
# Terminal 1 - GPU 0
DEVICE=0 python transformers/run_eval.py --model_id "openai/whisper-large-v3" ...

# Terminal 2 - GPU 1
DEVICE=1 python nemo_asr/run_eval.py --model_id "nvidia/canary-1b-v2" ...
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.5 GiB
```

**Solutions:**
1. Reduce `--batch_size`:
   ```bash
   --batch_size 4  # Try halving the default
   ```
2. Use gradient checkpointing (Whisper):
   ```bash
   --torch_compile  # Can reduce memory usage
   ```
3. Switch to a smaller model variant

### Model Download Issues

**Symptoms:**
```
HTTPError: 403 Forbidden
```

**Solutions:**
1. Authenticate with Hugging Face:
   ```bash
   huggingface-cli login
   ```
2. Check disk space:
   ```bash
   df -h  # Ensure >50GB free
   ```
3. Manually download model:
   ```bash
   huggingface-cli download openai/whisper-large-v3
   ```

### NeMo CUDA Version Issues

**Symptoms:**
```
RuntimeError: CUDA kernel launch failed
```

**Solution:**
NeMo requires CUDA 12.6+. Check version:
```bash
nvidia-smi  # Look for "CUDA Version: 12.6"
```

If version is too old, upgrade CUDA toolkit or use Whisper models instead.

### API Rate Limiting (ElevenLabs)

**Symptoms:**
```
HTTPError: 429 Too Many Requests
```

**Solution:**
Reduce concurrency:
```bash
python api/run_eval.py --max_workers 10  # Reduced from default 50
```

### Voxtral Evaluation Fails

**Symptoms:**
Model may not support audio input or requires special configuration.

**Solution:**
Check the [model card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) for audio input requirements. If unsupported, exclude from batch evaluation by commenting out in `evaluate_chilean_asr.sh`.

---

## Model-Specific Notes

### Whisper (OpenAI)
- Natively supports 99 languages including Spanish
- No special prompting required
- Best overall accuracy for Chilean Spanish in most cases

### Canary (NVIDIA)
- Multilingual model trained on English, Spanish, German, French
- Optimized for low-latency inference
- May require language hints for best results

### Parakeet (NVIDIA)
- Lightweight, fast model
- Good for real-time applications
- Trade-off: Speed vs. accuracy

### Phi-4 Multimodal (Microsoft)
- **Requires explicit Spanish prompt**: `"Transcribe el audio a texto en español."`
- Multimodal model (can handle images + audio)
- Largest model (14B parameters) - needs 24GB+ VRAM

### Voxtral (Mistral)
- Relatively new ASR model from Mistral AI
- May require special handling - check if it supports direct audio input
- If evaluation fails, verify model card for audio input format

### ElevenLabs Scribe
- **API-based** - requires internet connection and valid API key
- **Costs apply** - check ElevenLabs pricing
- High accuracy but slower than local models due to network latency
- Good for comparing against commercial solutions

---

## Advanced Usage

### Evaluate on a Subset of Data

Test quickly with a limited number of samples:

```bash
python transformers/run_eval.py \
    --model_id "openai/whisper-large-v3" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --max_eval_samples 100 \  # Only evaluate 100 samples
    --device 0
```

### Using Streaming Mode

For very large datasets, stream instead of downloading:

```bash
python transformers/run_eval.py \
    --model_id "openai/whisper-large-v3" \
    --dataset_path "astroza" \
    --dataset "es-cl-asr-test-only" \
    --streaming \  # Remove --no-streaming flag
    --device 0
```

**Note**: Streaming may be slower but uses less disk space.

### Custom Warmup Steps

Adjust the number of warmup iterations before timed evaluation:

```bash
python transformers/run_eval.py \
    --warmup_steps 20 \  # Default: 10
    ...
```

---

## Citation

If you use this evaluation framework or results, please cite both the Chilean Spanish dataset and the original Open ASR Leaderboard:

```bibtex
@misc{astroza2024chilean,
  title={Chilean Spanish ASR Test Dataset},
  author={Astroza},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/astroza/es-cl-asr-test-only}}
}

@misc{open-asr-leaderboard,
  title={Open Automatic Speech Recognition Leaderboard},
  author={Srivastav, Vaibhav and Majumdar, Somshubra and Koluguri, Nithin and Moumen, Adel and Gandhi, Sanchit and Hugging Face Team and Nvidia NeMo Team},
  year={2023},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/spaces/hf-audio/open_asr_leaderboard}}
}
```

---

## Contributing

This repository is a specialized fork focused on Chilean Spanish ASR evaluation. For contributions:

1. **Bug fixes & improvements**: Open an issue or pull request
2. **Adding new models**: Ensure they support Spanish and follow the existing evaluation structure
3. **General ASR leaderboard features**: Contribute to the upstream [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)

---

## License

This repository maintains the same license as the original [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard). It is provided as-is for research and evaluation purposes.

---

## Acknowledgments

- **Hugging Face, NVIDIA NeMo, and the Open ASR Leaderboard contributors** for the original evaluation framework
- **Astroza** for creating and sharing the Chilean Spanish ASR test dataset
- All model developers (OpenAI, NVIDIA, Microsoft, Mistral AI, ElevenLabs) for their ASR models

---

## Support

For issues specific to:
- **This Chilean Spanish adaptation**: Open an issue in this repository
- **Original Open ASR Leaderboard framework**: Visit [huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard)
- **Dataset issues**: Contact the [dataset author](https://huggingface.co/datasets/astroza/es-cl-asr-test-only)
- **Model-specific problems**: Check the respective model cards on Hugging Face
