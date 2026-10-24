"""Multilingual ASR evaluation for ARTPARK-IISc SraVaani-1.0.

SraVaani-1.0 (https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) is a ~430M
FastConformer with a hybrid TDT-CTC decoder, shipped as TorchScript graphs
behind a `trust_remote_code` wrapper. It needs no NeMo install, and it takes no
language conditioning — the script is inferred from the audio — so `--language`
only selects the normalizer and scorer.

The repo is **gated**: the job needs an HF_TOKEN whose account has accepted the
model's terms.

Batching note: the exported encoder does not length-mask its attention, so a
clip's encoder output shifts when it is padded up to a much longer neighbour
(measured: ~10% of the activation scale on a 2s clip padded to 15s). The decoded
text is far less sensitive — on 128 Monsoon Hindi clips, batch_size=32 differed
from batch_size=1 by 0.35% WER (121/128 transcripts identical). Batches run in
natural dataset order, like every other script here, so RTFx stays comparable
across models; use --batch_size=1 for exact model-card fidelity.
"""

import argparse
import json
import os
import re
import time

import evaluate
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import AutoModel

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs

wer_metric = evaluate.load("wer")

SAMPLING_RATE = 16_000

# Languages advertised on the model card's metadata. The card lists 63 Indian
# languages and dialects in total; these are the ones with ISO codes attached.
# Urdu and Kashmiri are explicitly not supported by this release.
SUPPORTED_LANGUAGES = {
    "hi", "kn", "ml", "te", "en", "gu", "pa", "or", "bn", "ta", "as", "sa", "ne", "mr",
}


def main(args):
    CONFIG_NAME = args.config_name  # None for single-config repos (e.g. VoiceArena/Monsoon_hi_test)
    SPLIT_NAME = args.split

    # Determine language for normalization: use --language if provided, otherwise
    # extract from config_name (e.g. "fleurs_hi") or, for single-config repos,
    # from the dataset name (e.g. "Monsoon_hi_test").
    if args.language:
        LANGUAGE = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        LANGUAGE = lang_match.group(1) if lang_match else "hi"

    if LANGUAGE not in SUPPORTED_LANGUAGES:
        # Advisory only: the model covers many more dialects than it tags, and
        # nothing in the inference path depends on the language.
        print(f"WARNING: '{LANGUAGE}' is not among SraVaani's tagged languages.")

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")

    # `torch_dtype` (not `dtype`) is what SraVaani's custom `from_pretrained`
    # reads; anything else is silently ignored and the model stays float32.
    model = AutoModel.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=args.dtype
    )
    model = model.to(device).eval()
    model._ensure_loaded(device)  # loads the TorchScript graphs + preprocessor
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    def load_eval_dataset():
        dataset = load_dataset(
            args.dataset,
            CONFIG_NAME,
            split=SPLIT_NAME,
            streaming=args.streaming,
            token=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            if args.streaming:
                dataset = dataset.take(args.max_eval_samples)
            else:
                dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
        return dataset

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [
            len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]
        ]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(
            batch, minibatch_size
        )

        # START TIMING
        on_cuda = device.type == "cuda"
        if on_cuda:
            torch.cuda.synchronize(device=device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        # `transcribe` batches internally; hand it the whole minibatch at once so
        # it does not re-split into its own default chunks of 8.
        pred_text = model.transcribe(audios, batch_size=minibatch_size)

        # END TIMING
        if on_cuda:
            end_event.record()
            torch.cuda.synchronize(device=device)
            runtime = start_event.elapsed_time(end_event) / 1000.0
        else:
            runtime = time.perf_counter() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        if "lattice" in batch:
            # Lattice reference (e.g. VoiceArena/Monsoon_hi_test): store the
            # lattice JSON-encoded in the reference field; scoring decodes it
            # and uses voi_oiwer (see normalizer/eval_utils.py).
            batch["references"] = [json.dumps(lat, ensure_ascii=False) for lat in batch["lattice"]]
        else:
            batch["references"] = batch["text"]  # raw; normalization applied at scoring time

        return batch

    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_eval_dataset()

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(
            warmup_dataset.map(
                benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]
            )
        )
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Reload the dataset for the timed run (resets the streaming pointer)
    dataset = load_eval_dataset()
    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }

    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Filter empty references (consistent with English pipeline)
    filtered = [
        (ref, pred, dur, time_s, fpath)
        for ref, pred, dur, time_s, fpath in zip(
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
            all_results["audio_filepath"],
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        (
            all_results["references"],
            all_results["predictions"],
            all_results["audio_length_s"],
            all_results["transcription_time_s"],
            all_results["audio_filepath"],
        ) = zip(*filtered)
        all_results = {k: list(v) for k, v in all_results.items()}

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset,
        CONFIG_NAME or "",
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    from normalizer.eval_utils import OIWER_LANGUAGES, score_oiwer

    if LANGUAGE in OIWER_LANGUAGES:
        # Lattice-based, orthography-aware scoring (voi_oiwer applies its own
        # normalization internally).
        manifest = [
            {"text": ref, "pred_text": pred}
            for ref, pred in zip(all_results["references"], all_results["predictions"])
        ]
        wer, _ins, _del, _sub = score_oiwer(manifest, OIWER_LANGUAGES[LANGUAGE])
        wer = round(100 * wer, 2)
    else:
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
        default="ARTPARK-IISc/SraVaani-1.0",
        help="Model identifier. Should be loadable with AutoModel + trust_remote_code.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'VoiceArena/Monsoon_hi_test'`",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name for the dataset. *E.g.* `'fleurs_hi'` for Hindi FLEURS. "
        "Omit for single-config dataset repos (e.g. 'VoiceArena/Monsoon_hi_test').",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'hi'). If not provided, extracted from config_name. "
        "Only selects the normalizer/scorer — the model is not language-conditioned.",
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
        default=32,
        help="Number of samples to go through each batch. Use 1 for exact model-card fidelity "
        "(the exported encoder is not length-masked; see the module docstring).",
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
        "--dtype",
        type=str,
        default="float32",
        help="Compute dtype. 'float32' (default) upcasts the shipped fp16 graphs, which is what "
        "the model does by default and is also faster here; 'float16' keeps them in fp16.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
