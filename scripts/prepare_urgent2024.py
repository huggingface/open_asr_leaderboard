# /// script
# dependencies = [
#   "datasets==3.6.0",
#   "librosa",
#   "soundfile==0.13.1",
# ]
# ///

import io
import re
import sys

import soundfile as sf
from datasets import Audio, Dataset, Features, Value, load_dataset


SOURCE_REPO_ID = "urgent-challenge/urgent2024_official"
# The "test_nonblind" config exposes its data under the "validation" split.
SOURCE_CONFIG = "test_nonblind"
SOURCE_SPLIT = "validation"

HUB_REPO_ID = "hf-audio/open-asr-leaderboard"
# Both configs are pushed to a single Hub PR: the first push opens it, the second
# pushes to its ref (refs/pr/N). Set to an existing PR ref to push to it instead.
HUB_REVISION = None


# Mapping from utterance id to fixed transcript; original URGENT2024 text in the comment
TRANSCRIPT_OVERRIDES = {
    # ### Identified fixes but not applied: https://github.com/huggingface/open_asr_leaderboard/pull/196#issuecomment-6023454846
    # # fileid_419:  doing our own share and our own part in this AI in this AI Revolution ution uh we sort of took our own little space in
    # "fileid_419": "doing our own share and our own part in this AI in this AI Revolution uh we sort of took our own little space in",
    # # fileid_422:  Intel this is significant you know industry gamechanging piece I actually just listened to some of your Clips this
    # "fileid_422": "Intel this is significant you know industry gamechanging piece I actually just listened to some of your clips this morning",
    # # fileid_832:  hopes in on this oate
    # "fileid_832": "In on this oblate spheroid?",
    # # fileid_841:  domestication is the driver of reduced brain sized and domesticated animals dog specifically but by doing a comparative philogenetic
    # "fileid_841": "domestication is the driver of reduced brain size in domesticated animals dogs specifically but by doing a comparative phylogenetic",
    # # fileid_841:  vaccine the thing that's going to keep us from getting whatever muuk is going around Corona virus
    # "fileid_841": "vaccine the thing that's going to keep us from getting whatever muck is going around Coronavirus",
    # # fileid_880:  disavowed um she's for the the green the green deal like1 trillion dollars over time effectively right talking about
    # "fileid_880": "disavowed um she's for the the green the green deal like a hundred trillion dollars over time effectively right talking about",
    # # fileid_885:  oan talk says wit it doesn't know the IV DV of electrostatics is this true wit
    # "fileid_885": "I was going to talk says: Witsit doesn't know the IV slash DV of electrostatics. Is this true, Witsit?",
    # # fileid_966:  I would rather see more of that. Yeah. Could it be another, when it comes to Atlus, could it be the Persona
    # "fileid_966": "Could it, yeah, could it be another, when it comes to Atlas, could it be the persona?",
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
        dataset_name = "urgent2024"
        features["snr_db"] = Value("float32")
    else:
        dataset_name = f"urgent2024_{subset}"

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
        text_orig = text
        if uid in TRANSCRIPT_OVERRIDES:
            text = TRANSCRIPT_OVERRIDES[uid]
        text = re.sub(r"[()]", "", text)
        if text != text_orig:
            print(f"Utterance {uid}:\n    {text_orig}\n -> {text}", file=sys.stderr)

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

    commit_info = ds.push_to_hub(
        HUB_REPO_ID,
        config_name=dataset_name,
        split="test",
        revision=HUB_REVISION,
        create_pr=HUB_REVISION is None,
        commit_message=f"Add {dataset_name}",
    )
    if HUB_REVISION is None:
        HUB_REVISION = commit_info.pr_revision
        print(f"Opened PR: {commit_info.pr_url}", file=sys.stderr)
