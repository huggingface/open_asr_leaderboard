import argparse
import os

import torch
from transformers import pipeline
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")


def main(args):
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        torch_dtype=torch.float16,
    )

    if asr_pipe.model.can_generate():
        gen_kwargs = {"max_new_tokens": 256}
        # for multilingual Whisper-checkpoints we see a definitive WER boost by setting the language and task args
        if getattr(asr_pipe.model.generation_config, "is_multilingual"):
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"
    else:
        gen_kwargs = None

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    def benchmark(batch):
        # get audio stats
        audio = [sample["array"] for sample in batch["audio"]]
        batch["audio_length"] = [len(sample) / 16_000 for sample in audio]
        minibatch_size = len(audio)

        # timing step
        start_time = time.time()
        result = asr_pipe(batch["audio"], generate_kwargs=gen_kwargs)
        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time"] = minibatch_size * [(time.time() - start_time) / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred["text"]) for pred in result]
        batch["references"] = batch["norm_text"]
        return batch

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]
    )

    all_results = {
        "audio_length": [],
        "transcription_time": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples"):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length"],
        transcription_time=all_results["transcription_time"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    print("WER:", wer, "%")

    transcription_time = sum(all_results["transcription_time"])
    audio_length = sum(all_results["audio_length"])
    rtfx = audio_length / transcription_time
    rtfx = round(rtfx, 2)
    print("RTFX:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
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
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
