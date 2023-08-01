#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--source="asr-wav2vec2-librispeech" \
    --speechbrain_pretrained_class_name="EncoderASR" \
	--dataset_path="librispeech_asr" \
	--dataset="clean" \
	--split="test" \
	--device=0 \
	--batch_size=32