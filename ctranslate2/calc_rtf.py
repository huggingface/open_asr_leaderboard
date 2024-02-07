import time
import librosa

from faster_whisper import WhisperModel

device = "cuda"
device_index = 0

models = [
    "guillaumekln/faster-whisper-tiny.en",
    "guillaumekln/faster-whisper-small.en",
    "guillaumekln/faster-whisper-base.en",
    "guillaumekln/faster-whisper-medium.en",
    "guillaumekln/faster-whisper-large-v1",
    "guillaumekln/faster-whisper-large-v2",
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
    asr_model = WhisperModel(
        model_size_or_path=model,
        device=device,
        device_index=device_index,
        compute_type="float16",
    )

    for i in range(3):
        print(f"outer_loop -> {i}")
        total_time = 0.0
        for _ in range(n_batches + warmup_batches):
            print(f"batch_num -> {_}")
            for idx in range(0, total_audio_len, chunk_len):
                chunk_signal = audio_dict["raw"][idx : idx + chunk_len * SAMPLE_RATE]
                start = time.time()
                segments, _ = asr_model.transcribe(chunk_signal, language="en")
                _ = [segment._asdict() for segment in segments]
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
