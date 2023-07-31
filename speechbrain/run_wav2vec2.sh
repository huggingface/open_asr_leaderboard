#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--source="asr-wav2vec2-librispeech" \
    --speechbrain_pretrained_class_name="EncoderASR" \
	--dataset="librispeech_asr" \
	--split="test" \
	--device=0 
