"""Run evaluation for vllm whisper models.""" ""
import argparse
import os
import time

import evaluate
from tqdm import tqdm
from normalizer import data_utils

from vllm import LLM
from vllm.sampling_params import SamplingParams

wer_metric = evaluate.load("wer")


def main(args) -> None:
    """Main function to run evaluation on a dataset."""

    device_id = "auto"
    if (args.device > 0):
        device_id = f"cuda:{args.device}"

    llm = LLM(
        model=args.model_id,
        max_model_len=448,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"audio": 1},
        kv_cache_dtype="fp8",
        device=device_id
    )

    def make_prompt(chunk, sr):
        return {
            "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            "multi_modal_data": {
                "audio": (chunk, sr),
            },
        }

    def process_vllm(batch):
        start_time = time.time()
        batch_size = len(batch["audio"])
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )
        # dataset resamples to 16kHz
        prompts = [ make_prompt(sample["array"], 16000.0) for sample in batch["audio"] ]
        outputs = llm.generate(prompts, sampling_params)
        # average transcription time over the whole batch
        batch["transcription_time_s"] = [ (time.time() - start_time) / batch_size ] * batch_size
        batch["predictions"] = [
            data_utils.normalizer("".join([output.outputs[0].text])).strip()
            for output in outputs
        ]
        batch["references"] = batch["norm_text"]
        return batch

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        process_vllm, batch_size=args.batch_size * 2, batched=True, remove_columns=["audio"]
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
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with faster-whisper",
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
        default=128,
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
