import time
import librosa

from faster_whisper import WhisperModel

device = "cuda"
device_index = 0

models = [
    "tiny.en",
    "small.en",
    "base.en",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3"
]

n_batches = 3
warmup_batches = 5

audio_file = "4469669.mp3"
max_len = 600  # 10 minutes


def pre_process_audio(audio_file, sr, max_len):
    _, _sr = librosa.load(audio_file, sr=sr)
    audio_len = int(max_len * _sr)
    audio_arr = _[:audio_len]
    return {"raw": audio_arr, "sampling_rate": _sr}, audio_len


audio_dict, audio_len = pre_process_audio(audio_file, 16000, max_len)

rtfs = []

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
            start = time.time()
            segments, _ = asr_model.transcribe(audio_dict["raw"], language="en")
            _ = [segment._asdict() for segment in segments]  # Iterate over segments to run inference
            end = time.time()
            if _ >= warmup_batches:
                total_time += end - start

        rtf = (total_time / n_batches) / (audio_len / 16000)
        rtfs.append(rtf)

    print(f"all RTFs: {model}: {rtfs}")
    rtf_val = sum(rtfs) / len(rtfs)
    print(f"avg. RTF: {model}: {rtf_val}")
