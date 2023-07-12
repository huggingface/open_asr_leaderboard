import argparse

from transformers import pipeline
from normalizer import EnglishTextNormalizer
from datasets import load_dataset, Audio
import evaluate

wer_metric = evaluate.load("wer")


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


def normalize(batch):
    batch["norm_text"] = normalizer(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args):
    asr_pipe = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )

    dataset = load_dataset(
        "esb/datasets",
        args.dataset_name,
        split=args.split,
        streaming=args.streaming,
        use_auth_token=True,
    )

    # Only uncomment for debugging
    dataset = dataset.take(args.max_eval_samples)

    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalize)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    predictions = []
    references = []

    # run streamed inference
    for out in asr_pipe(data(dataset), batch_size=args.batch_size):
        predictions.append(normalizer(out["text"]))
        references.append(out["reference"][0])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
            "can be found at `https://huggingface.co/datasets/esb/datasets`"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()

    main(args)

