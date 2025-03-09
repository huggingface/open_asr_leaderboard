import io

from datasets import load_dataset, Audio
import librosa
from normalizer import EnglishTextNormalizer

from .eval_utils import read_manifest, write_manifest


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )

normalizer = EnglishTextNormalizer()


def normalize(sample):
    sample["original_text"] = get_text(sample)
    sample["norm_text"] = normalizer(sample["original_text"])
    # Failover logic for dataset normalization
    if 'array' not in sample['audio'] and 'bytes' in sample['audio']:
        audio_file = io.BytesIO(sample['audio']['bytes'])
        audio_array, _ = librosa.load(audio_file, sr=16000)
        sample['audio']['array'] = audio_array
    return sample


def load_data(args):
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )

    return dataset

def prepare_data(dataset):
    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalize)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])
    return dataset


