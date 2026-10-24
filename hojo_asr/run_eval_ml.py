import argparse
import os
import torch
from hojo_asr import HOJO_ASR
import evaluate
from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs
import time
from tqdm import tqdm
from datasets import load_dataset, Audio

wer_metric = evaluate.load("wer")


def main(args):
    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split
    
    # Extract language from config_name if not provided
    if args.language:
        LANGUAGE = args.language
    else:
        try:
            LANGUAGE = CONFIG_NAME.split("_", 1)[1]
        except IndexError:
            LANGUAGE = "en"

    model = HOJO_ASR.load_model(args.model_id, device=args.device)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        # START TIMING
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        results = model.run_infer(audios, batch_size=args.batch_size)

        # Extract text predictions
        pred_text = [val["text"] for val in results]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Get references from the dataset
        batch["references"] = batch["text"]

        # normalize transcriptions with English normalizer
        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["text"]  # raw; normalization applied at scoring time

        return batch

    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )

    # Resample to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

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

    # Filter empty references (consistent with English pipeline)
    filtered = [
        (ref, pred, dur, time_s)
        for ref, pred, dur, time_s in zip(
            all_results["references"], all_results["predictions"],
            all_results["audio_length_s"], all_results["transcription_time_s"]
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        all_results["references"], all_results["predictions"], all_results["audio_length_s"], all_results["transcription_time_s"] = zip(*filtered)
        all_results = {k: list(v) for k, v in all_results.items()}

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset,
        CONFIG_NAME,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_refs = [data_utils.ml_normalizer(r, lang=LANGUAGE) for r in all_results["references"]]
    norm_preds = [data_utils.ml_normalizer(p, lang=LANGUAGE) for p in all_results["predictions"]]
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
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
        "--dataset",
        type=str,
        required=True,
        help=(
            "Hub id (e.g. 'nithinraok/asr-leaderboard-datasets') or local data root "
            "containing <corpus>/<lang>_test.parquet "
            "(e.g. '.../asr-leaderboard-datasets/data')."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name for the dataset. *E.g.* `'fleurs_en'` for English FLEURS.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'de'). If not provided, extracted from config_name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'` for the test split.",
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
        help="Number of samples to go through each batch.",
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
    parser.set_defaults(streaming=False)
    args = parser.parse_args()

    main(args)
