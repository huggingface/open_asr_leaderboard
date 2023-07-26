#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--model_id="openai/whisper-tiny" \
	--dataset_path="librispeech_asr" \
	--dataset="other" \
	--split="test" \
	--batch_size=32 \
	--device=0 
