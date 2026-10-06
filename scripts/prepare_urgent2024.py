# /// script
# dependencies = [
#   "datasets==3.6.0",
#   "librosa",
#   "soundfile==0.13.1",
# ]
# ///

import io
import sys

import soundfile as sf
from datasets import Audio, Dataset, Features, Value, load_dataset


SOURCE_REPO_ID = "urgent-challenge/urgent2024_official"
# The "test_nonblind" config exposes its data under the "validation" split.
SOURCE_CONFIG = "test_nonblind"
SOURCE_SPLIT = "validation"

HUB_REPO_ID = "hf-audio/open-asr-leaderboard"
HUB_REVISION = "main"


# Mapping from utterance id to fixed transcript; original URGENT2024 text in the comment
TRANSCRIPT_OVERRIDES = {
}


def to_wav_bytes(array, sampling_rate):
    buf = io.BytesIO()
    sf.write(buf, array, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


src = load_dataset(SOURCE_REPO_ID, SOURCE_CONFIG, split=SOURCE_SPLIT)

for subset in ["noisy", "clean"]:
    features = {
        "dataset": Value("string"),
        "audio": Audio(),
        "text": Value("string"),
        "id": Value("string"),
        "audio_length_s": Value("float32"),
    }

    if subset == "noisy":
        dataset_name = "urgent2024_nonblind"
        features["snr_db"] = Value("float32")
    else:
        dataset_name = f"urgent2024_nonblind_{subset}"

    audio_column = f"{subset}_audio"

    ids = []
    texts = []
    audio_field = []
    durations = []
    snrs = []
    for row in src:
        audio = row[audio_column]
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        uid = row["id"]
        text = row["transcript"]
        if uid in TRANSCRIPT_OVERRIDES:
            print(
                f"Utterance {uid}:\n    {text}\n -> {TRANSCRIPT_OVERRIDES[uid]}",
                file=sys.stderr,
            )
            text = TRANSCRIPT_OVERRIDES[uid]

        ids.append(uid)
        texts.append(text)
        audio_field.append(
            {"bytes": to_wav_bytes(array, sampling_rate), "path": f"{uid}.wav"}
        )
        durations.append(len(array) / sampling_rate)
        snrs.append(row.get("snr_dB"))

    for uid in TRANSCRIPT_OVERRIDES:
        if uid not in ids:
            print(
                f"Warning: TRANSCRIPT_OVERRIDES key {repr(uid)} does not match any of the dataset IDs",
                file=sys.stderr,
            )

    rows = {
        "dataset": [dataset_name] * len(ids),
        "audio": audio_field,
        "text": texts,
        "id": ids,
        "audio_length_s": durations,
    }
    if "snr_db" in features:
        rows["snr_db"] = snrs
    ds = Dataset.from_dict(rows, features=Features(features))
    ds = ds.sort("audio_length_s", reverse=True)

    ds.push_to_hub(
        HUB_REPO_ID, config_name=dataset_name, split="test", revision=HUB_REVISION
    )
