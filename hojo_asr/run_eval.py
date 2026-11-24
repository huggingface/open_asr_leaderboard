import json
import argparse
import os
import torch
from datasets import load_dataset
from hojo_asr import HOJO_ASR
import evaluate
from normalizer import data_utils
from normalizer import voice_code_bench
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")

def main(args):
    # VoiceCodeBench keeps its raw text and every recording; its WER and CTEM are computed from the manifest.
    is_vcb = voice_code_bench.is_voice_code_bench(args.dataset_path)
    # Load hojo model
    model = HOJO_ASR.load_model(args.model_id, device=args.device)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    if is_vcb:
        # run_infer takes its generation limit from the model config (about 200 tokens by default), which cuts off
        # VoiceCodeBench's 35-123 s recordings, so raise it for this dataset only.
        model.config.generate["max_new_tokens"] = args.max_new_tokens

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset, normalize_text=not is_vcb)

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]] 
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        # START TIMING
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if is_vcb:
            # The feature extractor truncates input at 40 s, so VoiceCodeBench's 35-123 s recordings are
            # transcribed in 30 s chunks and the pieces are joined per recording.
            size = 30 * 16000
            pieces, owners = [], []
            for i, a in enumerate(audios):
                for s in range(0, max(len(a), 1), size):
                    pieces.append(a[s:s + size])
                    owners.append(i)
            chunk_texts = [val["text"] for val in model.run_infer(pieces, batch_size=args.batch_size)]
            joined = [[] for _ in audios]
            for o, t in zip(owners, chunk_texts):
                joined[o].append(t)
            results = [{"text": " ".join(x for x in j if x)} for j in joined]
        else:
            results = model.run_infer(audios, batch_size=args.batch_size)

        # Extract text predictions
        pred_text = [val["text"] for val in results]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time

        return batch

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)
    metadata_keys = (
        voice_code_bench.MANIFEST_KEYS if is_vcb
        else data_utils.CHUNK_METADATA_KEYS if is_chunked else []
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }
    all_results.update({key: [] for key in metadata_keys})
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
        audio_filepaths=all_results["audio_filepath"],
        extra_fields={key: all_results[key] for key in metadata_keys},
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    if is_vcb:
        # The WER comes from score_results with the other datasets, the CTEM from the grading job.
        print("RTFx:", round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2))
        return

    if is_chunked:
        sessions = data_utils.merge_chunked_manifest(data_utils.read_manifest(manifest_path))
        references = [session["text"] for session in sessions]
        predictions = [session["pred_text"] for session in sessions]
    else:
        references = all_results["references"]
        predictions = all_results["predictions"]

    norm_refs = [data_utils.normalizer(r) for r in references]
    norm_preds = [data_utils.normalizer(p) for p in predictions]
    wer = wer_metric.compute(
        references=norm_refs, predictions=norm_preds
    )
    wer = round(100 * wer, 2)

    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with qwen_asr",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/open-asr-leaderboard",
        help="Dataset path. By default, it is `hf-audio/open-asr-leaderboard`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/hf-audio/open-asr-leaderboard`",
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
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )

    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
