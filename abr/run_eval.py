import argparse
import os
import time

import torch
import evaluate
import numpy as np
from normalizer import data_utils, cuda_sync
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

wer_metric = evaluate.load("wer")


def main(args):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_id, trust_remote_code=True
    ).cuda()
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).cuda()
    model.eval()  # Set model to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    def get_sub_batch_output(sub_batch):
        """Get output from model on sub batch."""

        features = feature_extractor(sub_batch, return_tensors="pt")
        inputs = features["input_features"]

        if inputs.shape[1] < 8:
            # Shortcut for inputs too short to process
            pred_text = ["" for _ in inputs]
        else:
            # Get output from model with inference mode
            with torch.inference_mode(): 
                outputs = model(inputs, mask=features["mask"])

            # Decode text
            pred_text = tokenizer.decode_from_logits(outputs["logits"], outputs["mask"])
        return pred_text

    def benchmark(batch):
        # Load audio inputs (preprocessing outside timed block)
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)

        # Divide data into sub-batches that maximize the total audio length
        # that can fit within the specified limit
        sort_idxs = np.argsort([len(x) for x in audios])
        all_out = [None for _ in audios]
        sorted_audios = [audios[i] for i in sort_idxs]
        sub_batch = []
        sub_batch_idxs = []  # Track which sorted indices are in sub_batch

        # START TIMING - CUDA sync for accurate GPU timing
        cuda_sync(args.device)  # ABR model runs on CUDA
        start_time = time.time()

        for i, audio in enumerate(sorted_audios):
            n_samples = len(audio) * (len(sub_batch) + 1)

            if n_samples >= args.subbatch_samples:
                # When we reach the size limit, get output from sub-batch
                pred_text = get_sub_batch_output(sub_batch)

                # Put sub-batch outputs back into the appropriate spots in the overall
                # batch
                for j in range(len(sub_batch)):
                    target_idx = sort_idxs[sub_batch_idxs[j]]
                    assert all_out[target_idx] is None
                    all_out[target_idx] = pred_text[j]

                sub_batch = []
                sub_batch_idxs = []

            sub_batch.append(audio)
            sub_batch_idxs.append(i)

        # Process any leftover items
        if sub_batch:
            pred_text = get_sub_batch_output(sub_batch)

            for j in range(len(sub_batch)):
                target_idx = sort_idxs[sub_batch_idxs[j]]
                assert all_out[target_idx] is None
                all_out[target_idx] = pred_text[j]

        # END TIMING - CUDA sync before measuring
        cuda_sync(args.device)
        runtime = time.time() - start_time

        assert all(x is not None for x in all_out)
        pred_text = all_out

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(
                range(min(num_warmup_samples, len(dataset)))
            )
        warmup_dataset = iter(
            warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True)
        )

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
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
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
        default=0,
        help="The device to run the pipeline on. 0 for the first GPU (default) and so on.",
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
        dest="streaming",
        action="store_true",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    parser.add_argument(
        "--subbatch_samples",
        type=int,
        default=int(1e6),
        help="Maximum number of audio samples per sub batch (set based on available GPU memory).",
    )
    args = parser.parse_args()

    main(args)
