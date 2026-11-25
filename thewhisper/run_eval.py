import argparse
import os
import re

import evaluate
import numpy as np
import torch
from huggingface_hub import get_safetensors_metadata
from normalizer import data_utils
from normalizer import voice_code_bench
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from elastic_models.transformers import WhisperForConditionalGeneration
from elastic_models.transformers.pipelines.asr_vad_chunked import TheStageASRPipelineVAD


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


# Input window per clip: (window in seconds, longest clip that goes to it); longer clips use --chunk_length.
WINDOW_RULE = ()
LOOP_UNITS_PER_S = 6.0


def window_for(duration_s, longest_window):
    for window, limit in WINDOW_RULE:
        if window < longest_window and duration_s <= limit:
            return window
    return longest_window


def set_window(model, window_s):
    positions = int(round(window_s * 50))  # 50 encoder positions per second of audio
    model.config.max_source_positions = positions
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    if encoder is not None and getattr(encoder, "config", None) is not None:
        encoder.config.max_source_positions = positions


def looks_looped(text, duration_s):
    # Hyphens and punctuation separate words, so "Do-Do-Do-..." counts as a repetition.
    units = [u for u in re.split(r"[\s\-_.,;:!?()\[\]\"\u2010-\u2015\u0964\u3001\u3002]+", text.lower()) if u]
    if duration_s > 0 and len(units) / duration_s > LOOP_UNITS_PER_S:
        return "rate"
    if len(units) >= 12 and len(set(units)) <= 0.3 * len(units):
        return "repeat"
    for n, repeats in ((1, 6), (2, 4), (3, 4)):
        for s in range(len(units) - n * repeats + 1):
            piece = units[s:s + n]
            if all(units[s + k * n:s + (k + 1) * n] == piece for k in range(1, repeats)):
                return "repeat"
    return None


def run_window(asr_pipeline, model, clips, window, longest_window, lang_ids, **kwargs):
    set_window(model, window)
    try:
        res = asr_pipeline(clips, chunk_length_s=window, lang_ids=lang_ids, **kwargs)
    finally:
        set_window(model, longest_window)
    return [res] if isinstance(res, dict) else res


def transcribe(asr_pipeline, model, audios, sampling_rate, longest_window, lang_ids, **kwargs):
    durations = [len(audio) / sampling_rate for audio in audios]
    windows = [window_for(d, longest_window) for d in durations]
    outputs = [None] * len(audios)

    def rerun(idx, **run_kwargs):
        res = run_window(
            asr_pipeline, model, [audios[i] for i in idx], longest_window, longest_window,
            [lang_ids[i] for i in idx], **run_kwargs,
        )
        for i, r in zip(idx, res):
            outputs[i] = r

    for window in sorted(set(windows), reverse=True):
        idx = [i for i, w in enumerate(windows) if w == window]
        res = run_window(
            asr_pipeline, model, [audios[i] for i in idx], window, longest_window,
            [lang_ids[i] for i in idx], **kwargs,
        )
        for i, r in zip(idx, res):
            outputs[i] = r

    # Looping outputs are decoded again in the longest window, then with repeated 3-grams forbidden.
    redo = [
        i for i in range(len(audios))
        if windows[i] < longest_window and looks_looped(outputs[i]["text"], durations[i])
    ]
    if redo:
        rerun(redo, **kwargs)
    redo = [i for i in range(len(audios)) if looks_looped(outputs[i]["text"], durations[i]) == "rate"]
    if redo:
        gen_kwargs = dict(kwargs.get("generate_kwargs") or {}, no_repeat_ngram_size=3)
        rerun(redo, **dict(kwargs, generate_kwargs=gen_kwargs))
    return outputs


def warmup_windows(asr_pipeline, model, sampling_rate, longest_window, batch_size, language, gen_kwargs):
    # Dummy audio for every window, independent of the dataset.
    rng = np.random.default_rng(0)
    for window in sorted({w for w, _ in WINDOW_RULE if w < longest_window} | {longest_window}):
        dummy = [
            (0.01 * rng.standard_normal(int(0.9 * window * sampling_rate))).astype(np.float32)
            for _ in range(batch_size)
        ]
        transcribe(
            asr_pipeline, model, dummy, sampling_rate, window, [language] * batch_size,
            batch_size=batch_size, generate_kwargs=gen_kwargs,
        )


def sort_by_duration(dataset):
    for column in ("audio_length_s", "duration"):
        if column in dataset.column_names:
            return dataset.sort(column, reverse=True)
    return dataset


def main(args):
    torch_dtype = torch.float16

    # TheStage AI compiled engines (TensorRT). The same revision pins the configs and the engines;
    # without it elastic_models falls back to the latest tag of the model repo.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        mode=args.mode,
        chunk_length=args.chunk_length,
        revision=args.revision,
        elastic_revision=args.revision,
    ).to(args.device)
    model.eval()
    # The encoder and decoder layers run as TensorRT engines rather than PyTorch parameters, so the total
    # number of parameters is read from the checkpoint of the same revision.
    num_params = sum(get_safetensors_metadata(args.model_id, revision=args.revision).parameter_count.values())
    print(f"Model size: {num_params / 1e9:.2f}B parameters")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        chunk_length=args.chunk_length,
        tokenizer=AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, use_fast=True),
    )
    sampling_rate = processor.feature_extractor.sampling_rate
    # TheStage AI ASR pipeline, as in run_eval_longform.py: log-Mel features are computed on the GPU; inputs
    # longer than the `chunk_length` window are split at pauses by Silero VAD and the pieces joined.
    cuda_device = torch.device("cuda", args.device)
    asr_pipeline = TheStageASRPipelineVAD(model, processor, cuda_device, feature_extractor_device=cuda_device)

    # The generation config of the checkpoint (suppress_tokens included) is used as shipped.
    # forced_decoder_ids would override the language passed to the pipeline below.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.cache_implementation = "flexi-static"
    gen_kwargs = {"num_beams": 1, "do_sample": False, "disable_compile": True}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(audio) / sampling_rate for audio in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        # START TIMING
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        outputs = transcribe(
            asr_pipeline, model, list(audios), sampling_rate, args.chunk_length, ["en"] * minibatch_size,
            batch_size=args.batch_size,
            generate_kwargs=gen_kwargs,
        )
        pred_text = [output["text"] for output in outputs]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
        return batch

    # VoiceCodeBench keeps its raw text (the entity judge needs it) and every recording, so that its coverage stays
    # complete; its WER and CTEM are computed from the manifest (see voice_code_bench.md).
    is_vcb = voice_code_bench.is_voice_code_bench(args.dataset_path)

    warmup_windows(asr_pipeline, model, sampling_rate, args.chunk_length, args.batch_size, "en", gen_kwargs)

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate, normalize_text=not is_vcb)

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
    dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate, normalize_text=not is_vcb)
    if not args.streaming:
        dataset = sort_by_duration(dataset)

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
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
    # Filtering of empty references is handled inside write_manifest.
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
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="TheStageAI/thewhisper-large-v3-turbo",
        help="Model identifier on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        required=True,
        help="Model repo revision (commit hash). Pins both the configs and the compiled engines.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="XL",
        choices=["S", "M", "L", "XL"],
        help="TheStage AI engine size.",
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=30,
        help="Input window of the compiled engines, in seconds.",
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
        help="Dataset name, e.g. `voxpopuli_cleaned_aa`.",
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
        help="The GPU to run on.",
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
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
