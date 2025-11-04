# NOTES
## common folders
- data (just samples)
- normalizer
    - dict of >1700 english abbreviations
    - data and eval utils
    - normalizer.py (this could be the most useful thing for us)
        - we will probably need to create a proper slovak normalizer
- requirements

## folders for specific models
- api (models accessed via API)
- ctranslate2 (faster_whisper)
- granite (IBM granite speech)
- kyutai (kyutai/stt-2.6b-en)
- liteASR (efficient-speech/lite-whisper*)
- moonshine (usefulsensors/moonshine-*)
- nemo_asr (nvidia parakeet, canary models)
- phi (microsoft/Phi-4-multimodal-instruct)
- speechbrain (speechbrain/asr-*)
- tensorrtllm (TensorRT‑LLM whisper models)
- transformers (whisper, facebook mms, wav2vec2, hubert, data2vec)

# TEST RUN
# run small test inference
```
cd transformers/
bash run_whisper_test.sh
```

# evaluate results
```
cd /home/vidogreq/open_asr_leaderboard/normalizer
python -c "import eval_utils; eval_utils.score_results('../transformers/results', 'openai/whisper-tiny.en')"
```

# result
```
Filtering models by id: openai/whisper-tiny.en
********************************************************************************
Results per dataset:
********************************************************************************
openai/whisper-tiny.en | hf-audio-esb-datasets-test-only-sorted_gigaspeech_test: WER = 5.29 %, RTFx = 51.75

********************************************************************************
Composite Results:
********************************************************************************
openai/whisper-tiny.en: WER = 5.29 %
openai/whisper-tiny.en: RTFx = 51.75
********************************************************************************
```