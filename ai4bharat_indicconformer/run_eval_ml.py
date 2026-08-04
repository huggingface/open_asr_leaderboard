"""Multilingual ASR evaluation for AI4Bharat IndicConformer-600M-Multi.

IndicConformer (https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
is a 600M hybrid CTC + RNNT Conformer covering the 22 scheduled Indian
languages. It ships as ONNX graphs plus a TorchScript preprocessor behind a
`trust_remote_code` wrapper, and is language-conditioned: `lang` selects the
vocabulary, the CTC language mask, and the per-language RNNT joint head
(`joint_post_net_<lang>.onnx`).

The repo is **gated**: the job needs an HF_TOKEN whose account has accepted the
model's terms. `from_pretrained` pulls the full ~2.5 GB snapshot.

There is no English support, so there is no English `run_eval.py` counterpart.

Batching: the shipped `model_onnx.py` decodes one utterance at a time. Its CTC
path is commented "currently no batching" and reads `logprobs[0]` while ignoring
`encoded_lengths`; its RNNT path carries a batch-1 decoder state throughout. The
repo also contains `model_onnx_1b_batched_rnnt.py`, but that module requires
`assets/rnnt_decoder_embed.onnx` and `assets/rnnt_decoder_rnn.onnx`, which this
600M repo does not ship (it targets the 1B variant), so it cannot be used here.
This script therefore transcribes per utterance, exactly as the model card does.
`--batch_size` only controls how many rows `datasets.map` groups per call, which
keeps the manifest and timing bookkeeping identical to the sibling scripts.
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

# The 22 scheduled Indian languages, as named by the per-language joint heads in
# the repo (assets/joint_post_net_<lang>.onnx). Note several are 3-letter codes.
SUPPORTED_LANGUAGES = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}


def main(args):
    CONFIG_NAME = args.config_name  # None for single-config repos (e.g. VoiceArena/Monsoon_hi_test)
    SPLIT_NAME = args.split

    # Determine language: use --language if provided, otherwise extract from
    # config_name (e.g. "fleurs_hi") or, for single-config repos, from the
    # dataset name (e.g. "Monsoon_hi_test").
    if args.language:
        LANGUAGE = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        LANGUAGE = lang_match.group(1) if lang_match else "hi"

    if LANGUAGE not in SUPPORTED_LANGUAGES:
        # Unlike the other models here, the language is not advisory: it picks
        # the vocabulary and the RNNT joint head, so an unknown code would raise
        # a KeyError deep inside the model.
        raise ValueError(
            f"IndicConformer does not support language '{LANGUAGE}'. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    # The model hardcodes `cuda` when available and onnxruntime's CUDA provider
    # defaults to GPU 0, so --device only picks the torch-side device. On a
    # multi-GPU box, set CUDA_VISIBLE_DEVICES to pin onnxruntime too.
    if args.device >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device={args.device} requested but CUDA is not available.")
        torch.cuda.set_device(args.device)
    on_cuda = args.device >= 0 and torch.cuda.is_available()

    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    print(f"Loaded IndicConformer; decoding={args.decoding}, language={LANGUAGE}")

    def transcribe(audios):
        """One utterance at a time — see the module docstring on batching."""
        preds = []
        for audio in audios:
            wav = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
            preds.append(model(wav, LANGUAGE, args.decoding))
        return preds

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
        if on_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        pred_text = transcribe(audios)

        # END TIMING
        if on_cuda:
            end_event.record()
            torch.cuda.synchronize()
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
        default="ai4bharat/indic-conformer-600m-multilingual",
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
        "Selects the vocabulary, CTC language mask and RNNT joint head.",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="rnnt",
        choices=["rnnt", "ctc"],
        help="Decoding head of this hybrid model. 'rnnt' (default) is normally the more "
        "accurate one; 'ctc' is a single forward pass and much faster.",
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
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on. "
        "onnxruntime always uses GPU 0; pin it with CUDA_VISIBLE_DEVICES on multi-GPU hosts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Rows grouped per datasets.map call. Inference is per utterance either way "
        "(the shipped model has no working batched path; see the module docstring).",
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
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
