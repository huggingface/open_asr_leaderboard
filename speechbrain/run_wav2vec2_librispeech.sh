#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

SOURCE="speechbrain/asr-wav2vec2-librispeech"


python run_eval.py \
	--source=$MODEL_ID \
    --speechbrain_pretrained_class_name="EncoderASR" \
	--dataset_path="librispeech_asr" \
	--dataset="clean" \
	--split="test" \
	--device=0 \
	--batch_size=32