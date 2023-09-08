"""Run evaluation for ctranslate2 whisper models."""""
import argparse
import os

import evaluate
from faster_whisper import WhisperModel
from tqdm import tqdm

from normalizer import data_utils

wer_metric = evaluate.load("wer")


def dataset_iterator(dataset) -> dict:
    """
    Iterate over the dataset and yield a dictionary with the audio and reference text.

    Args:
        dataset: dataset to iterate over

    Returns:
        dictionary: {"audio": audio, "reference": reference}
    """
    for item in dataset:
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args) -> None:
    """Main function to run evaluation on a dataset."""
    asr_model = WhisperModel(
        model_size_or_path=args.model_id,
        compute_type="float16",
        device="cuda",
        device_index=args.device
    )

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    predictions = []
    references = []

    # Run inference
    for batch in tqdm(dataset_iterator(dataset), desc=f"Evaluating {args.model_id}"):
        segments, _ = asr_model.transcribe(batch["array"], language="en")
        outputs = [segment._asdict() for segment in segments]
        predictions.extend(
            data_utils.normalizer(
                "".join([segment["text"] for segment in outputs])
            ).strip()
        )
        references.extend(batch["reference"][0])

    # Write manifest results
    manifest_path = data_utils.write_manifest(
        references, predictions, args.model_id, args.dataset_path, args.dataset, args.split
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        '--dataset_path', type=str, default='esb/datasets', help='Dataset path. By default, it is `esb/datasets`'
    )
    parser.add_argument(
        "--dataset",
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
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
