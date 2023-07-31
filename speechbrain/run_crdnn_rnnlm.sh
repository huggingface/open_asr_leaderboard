#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--source="asr-crdnn-rnnlm-librispeech" \
    --speechbrain_pretrained_class_name="EncoderDecoderASR" \
	--dataset_path="librispeech_asr" \
	--dataset="other" \
	--split="test" \
	--device=0 \
	--batch_size=4