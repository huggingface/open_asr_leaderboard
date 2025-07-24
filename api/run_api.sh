#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

export OPENAI_API_KEY="your_api_key"
export ASSEMBLYAI_API_KEY="your_api_key"
export ELEVENLABS_API_KEY="your_api_key"
export REVAI_API_KEY="your_api_key"
export GOOGLE_API_KEY="your_api_key"

MODEL_IDs=(
    "openai/gpt-4o-transcribe"
    "openai/gpt-4o-mini-transcribe"
    "openai/whisper-1"
    "assembly/best"
    "elevenlabs/scribe_v1"
    "revai/machine" # please use --use_url=True
    "revai/fusion" # please use --use_url=True
    "speechmatics/enhanced"
    "google/gemini-2.5-pro"
)

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.clean" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "librispeech" \
        --split "test.other" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --model_name ${MODEL_ID}

    python run_eval.py \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Yoruba
    python run_eval.py \
        --dataset_path="multilingual-datasets/yoruba" \
        --dataset="yoruba_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Igbo  
    python run_eval.py \
        --dataset_path="multilingual-datasets/igbo" \
        --dataset="igbo_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Hausa
    python run_eval.py \
        --dataset_path="multilingual-datasets/hausa" \
        --dataset="hausa_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - French
    python run_eval.py \
        --dataset_path="multilingual-datasets/french" \
        --dataset="french_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Amharic
    python run_eval.py \
        --dataset_path="multilingual-datasets/amharic" \
        --dataset="amharic_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Malagasy
    python run_eval.py \
        --dataset_path="multilingual-datasets/malagasy" \
        --dataset="malagasy_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Pidgin
    python run_eval.py \
        --dataset_path="multilingual-datasets/pidgin" \
        --dataset="pidgin_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Multilingual datasets - Swahili
    python run_eval.py \
        --dataset_path="multilingual-datasets/swahili" \
        --dataset="swahili_speech" \
        --split="test" \
        --model_name ${MODEL_ID}

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
