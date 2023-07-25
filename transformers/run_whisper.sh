#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--model_id="openai/whisper-tiny" \
	--dataset="librispeech_asr" \
	--split="test" \
	--device=0 
