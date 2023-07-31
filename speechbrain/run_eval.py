""" 
Script to evaluate a pretrained SpeechBrain model on ESB.

Authors
* Adel Moumen 2023
"""
import argparse

import evaluate
from normalizer import data_utils

import speechbrain.pretrained as pretrained
import os 

def get_model(speechbrain_repository, speechbrain_pretrained_class_name, savedir=None, **kwargs):

    run_opt_defaults = {
        "device": "cpu",
        "data_parallel_count": -1,
        "data_parallel_backend": False,
        "distributed_launch": False,
        "distributed_backend": "nccl",
        "jit_module_keys": None,
    }

    run_opts = {**run_opt_defaults, **kwargs}
    
    if run_opts["device"] == -1:
        run_opts["device"] = "cpu"
    else:
        run_opts["device"] = f"cuda:{run_opts['device']}"

    kwargs = {
        "source": f"speechbrain/{speechbrain_repository}",
        "savedir": f"pretrained_models/{speechbrain_repository}",
    }

    if savedir is not None:
        kwargs["savedir"] = savedir

    try:
        model_class = getattr(pretrained, speechbrain_pretrained_class_name)
    except AttributeError:
        raise AttributeError(f"SpeechBrain Pretrained class: {speechbrain_pretrained_class_name} not found in pretrained.py")

    return model_class


def dataset_iterator(dataset):
    for i, item in enumerate(dataset):
        yield {
            **item["audio"],
            "reference": item["norm_text"],
            "audio_filename": f"file_{i}",
            "sample_rate": 16_000,
            "sample_id": i,
        }

wer_metric = evaluate.load("wer")

def main(args):
    asr_model = get_model(args.source, args.speechbrain_pretrained_class_name, device=args.device)

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    predictions = []
    references = []

    # run streamed inference
    for out in asr_pipe(dataset_iterator(dataset), batch_size=args.batch_size):
        predictions.append(data_utils.normalizer(out["text"]))
        references.append(out["reference"][0])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="SpeechBrain model repository. E.g. `asr-crdnn-rnnlm-librispeech`",
    )

    parser.add_argument(
        "--speechbrain_pretrained_class_name",
        type=str,
        required=True,
        help="SpeechBrain pretrained class name. E.g. `EncoderASR`",
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
    parser.set_defaults(streaming=True)

    main(args)

