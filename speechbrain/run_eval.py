"""Script to evaluate a pretrained SpeechBrain model from the 🤗 Hub.

Authors
* Adel Moumen 2023 <adel.moumen@univ-avignon.fr>
* Sanchit Gandhi 2024 <sanchit@huggingface.co>
"""
import argparse
import time

import evaluate
from normalizer import data_utils
from tqdm import tqdm
import torch
import speechbrain.inference.ASR as ASR
from speechbrain.utils.data_utils import batch_pad_right
import os

def get_model(
    speechbrain_repository: str,
    speechbrain_pretrained_class_name: str,
    **kwargs,
):
    """Fetch a pretrained SpeechBrain model from the SpeechBrain 🤗 Hub.

    Arguments
    ---------
    speechbrain_repository : str
        The name of the SpeechBrain repository to fetch the pretrained model from. E.g. `asr-crdnn-rnnlm-librispeech`.
    speechbrain_pretrained_class_name : str
        The name of the SpeechBrain pretrained class to fetch. E.g. `EncoderASR`.
        See: https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/pretrained/interfaces.py
    **kwargs
        Additional keyword arguments to pass to override the default run options of the pretrained model.

    Returns
    -------
    SpeechBrain pretrained model
        The Pretrained model.

    Example
    -------
    >>> from open_asr_leaderboard.speechbrain.run_eval import get_model
    >>> model = get_model("asr-crdnn-rnnlm-librispeech", "EncoderASR", device="cuda:0")
    """

    run_opt_defaults = {
        "device": "cpu",
        "data_parallel_count": -1,
        "data_parallel_backend": False,
        "distributed_launch": False,
        "distributed_backend": "nccl",
        "jit_module_keys": None,
    }

    run_opts = {**run_opt_defaults, **kwargs}

    kwargs = {
        "source": f"{speechbrain_repository}",
        "savedir": f"pretrained_models/{speechbrain_repository}",
        "run_opts": run_opts,
    }

    try:
        model_class = getattr(ASR, speechbrain_pretrained_class_name)
    except AttributeError:
        raise AttributeError(
            f"SpeechBrain Pretrained class: {speechbrain_pretrained_class_name} not found in pretrained.py"
        )

    return model_class.from_hparams(**kwargs)


def main(args):
    """Run the evaluation script."""
    if args.device == -1:
        device = "cpu"
    else:
        device = f"cuda:{args.device}"

    model = get_model(
        args.source, args.speechbrain_pretrained_class_name, device=device
    )

    def benchmark(batch):
        # Load audio inputs
        audios = [torch.from_numpy(sample["array"]) for sample in batch["audio"]]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        audios, audio_lens = batch_pad_right(audios)
        audios = audios.to(device)
        audio_lens = audio_lens.to(device)
        predictions, _ = model.transcribe_batch(audios, audio_lens)

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in predictions]
        batch["references"] = batch["norm_text"]
        return batch


    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.source,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


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
        "--dataset_path",
        type=str,
        default="hf-audio/esb-datasets-test-only-sorted",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
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
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args)
