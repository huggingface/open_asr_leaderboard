import time
import torch
import librosa

from transformers import pipeline

device = "cuda:0"

models = [
    "facebook/hubert-large-ls960-ft",
    "facebook/hubert-xlarge-ls960-ft",
    "patrickvonplaten/hubert-xlarge-ls960-ft-4-gram",
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60-self",
    "facebook/wav2vec2-large-robust-ft-libri-960h",
    "facebook/wav2vec2-conformer-rel-pos-large-960h-ft",
    "facebook/wav2vec2-conformer-rope-large-960h-ft",
    "openai/whisper-tiny.en",
    "openai/whisper-small.en",
    "openai/whisper-base.en",
    "openai/whisper-medium.en",
    "openai/whisper-large",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "facebook/mms-1b-all",
    "facebook/mms-1b-fl102",
]

n_batches = 3
warmup_batches = 3

audio_file = "../data/sample_4469669.wav"
chunk_len = 30
total_audio_len = 600  # 10 min
SAMPLE_RATE = 16000
total_chunks = int(total_audio_len / chunk_len)


def pre_process_audio(audio_file, sr, max_len):
    _, _sr = librosa.load(audio_file, sr=sr)
    audio_len = int(max_len * _sr)
    audio_arr = _[:audio_len]
    return {"raw": audio_arr, "sampling_rate": _sr}, audio_len


audio_dict, audio_len = pre_process_audio(audio_file, SAMPLE_RATE, total_audio_len)

rtfxs = []

for model in models[:1]:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        torch_dtype=torch.float16,
        batch_size=1,
    )

    for i in range(3):
        print(f"outer_loop -> {i}")
        total_time = 0.0
        for _ in range(n_batches + warmup_batches):
            print(f"batch_num -> {_}")
            for idx in range(0, total_audio_len, chunk_len):
                chunk_signal = audio_dict["raw"][idx : idx + chunk_len * SAMPLE_RATE]
                start = time.time()
                transcription = pipe(
                    {
                        "raw": chunk_signal,
                        "sampling_rate": audio_dict["sampling_rate"],
                    },
                    chunk_length_s=chunk_len,
                )
                end = time.time()
                if _ >= warmup_batches:
                    total_time += end - start
        
        avg_time_per_chunk = total_time / total_chunks
        rtf = (avg_time_per_chunk / n_batches) / (chunk_len)
        rtfx = float(1 / rtf)
        rtfxs.append(rtfx)

    print(f"all RTFs: {model}: {rtfxs}")
    rtfx_val = int(sum(rtfxs) / len(rtfxs))
    print(f"avg. RTFX: {model}: {rtfx_val}")
