import time
from transformers import pipeline
import librosa

device = "cuda:0"
model = "openai/whisper-tiny.en"
n_batches = 5
warmup_batches = 3
pipe = pipeline("automatic-speech-recognition", model=model, device=device)

audio_file = "4469669.mp3"
max_len = 600 #10 minutes

def pre_process_audio(audio_file, sr, max_len):
    _, sr = librosa.load(audio_file, sr=sr)
    _mono = librosa.to_mono(_)
    audio_len = int(max_len * sr)
    audio_arr = _mono[audio_len]
    return {"raw": audio_arr, "sampling_rate" = 16000}, audio_len

audio_dict, audio_len = pre_process_audio(audio_file)

rtfs = []
for i in range(3):
    for _ in range(n_batchs + warmup_batches):
        start = time.time()
        transcription = pipe(audio, chunk_length_s=15)
        end = time.time()
        if _ >= warmup_batches:
            total_time += end - start
        
        rtf = (total_time/nbatches) / (audio_len / 16000)
    rtfs.append(rtf)

print(f'RTF: {rtfs}')
rtf_val = sum(rtfs)/len(rtfs)
print(f'avg. RTF: {rtf_val}')