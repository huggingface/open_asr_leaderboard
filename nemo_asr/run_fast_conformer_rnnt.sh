#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

python run_eval.py \
	--model_id="nvidia/stt_en_conformer_transducer_large" \
	--dataset_path="librispeech_asr" \
	--dataset="other" \
	--split="test" \
	--device=0 \
	--batch_size=32 \
	--max_eval_samples=-1
