#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--model_id="facebook/wav2vec2-base-960h" \
	--dataset="librispeech" \
	--split="test.other" \
	--device=0 
