import time
from transformers import pipeline
import librosa

device = "cpu"
model = "openai/whisper-tiny.en"
n_batches = 5
warmup_batches = 3
pipe = pipeline("automatic-speech-recognition", model=model, device=device)

audio_file = "4469669.mp3"
max_len = 600 #10 minutes

def pre_process_audio(audio_file, sr, max_len):
    _, _sr = librosa.load(audio_file, sr=sr)
    audio_len = int(max_len * _sr)
    audio_arr = _[:audio_len]
    return {"raw": audio_arr, "sampling_rate": _sr}, audio_len

audio_dict, audio_len = pre_process_audio(audio_file, 16000, max_len)

rtfs = []
for i in range(3):
    print(f'outer_loop -> {i}')
    total_time = 0.0
    for _ in range(n_batches + warmup_batches):
        print(f"batch_num -> {_}")
        start = time.time()
        transcription = pipe({"raw":audio_dict["raw"], "sampling_rate": audio_dict["sampling_rate"]}, chunk_length_s=10)
        end = time.time()
        if _ >= warmup_batches:
            total_time += end - start
        
    rtf = (total_time/n_batches) / (audio_len / 16000)
    rtfs.append(rtf)

print(f'all RTFs: {rtfs}')
rtf_val = sum(rtfs)/len(rtfs)
print(f'avg. RTF: {rtf_val}')
