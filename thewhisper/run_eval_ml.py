import argparse
import json
import os
import re

import evaluate
import torch
from datasets import Audio, load_dataset
from huggingface_hub import get_safetensors_metadata
from normalizer import data_utils
from normalizer.eval_utils import OIWER_LANGUAGES, normalize_compound_pairs, score_oiwer
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from elastic_models.transformers import WhisperForConditionalGeneration
from elastic_models.transformers.pipelines.asr_vad_chunked import TheStageASRPipelineVAD


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


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

    CONFIG_NAME = args.config_name  # None for single-config dataset repos
    SPLIT_NAME = args.split

    # Language for decoding and normalization: --language if given, otherwise taken from the
    # config name (e.g. "fleurs_de")
    if args.language is not None:
        norm_language = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        norm_language = lang_match.group(1) if lang_match else "en"
        print(f"Language not specified, extracted '{norm_language}' from '{source}'")

    # The generation config of the checkpoint (suppress_tokens included) is used as shipped.
    # forced_decoder_ids would override the language passed to the pipeline below.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.cache_implementation = "flexi-static"
    gen_kwargs = {"num_beams": 1, "do_sample": False, "disable_compile": True}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    # Load dataset
    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

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

        outputs = asr_pipeline(
            list(audios),
            batch_size=args.batch_size,
            chunk_length_s=args.chunk_length,
            generate_kwargs=gen_kwargs,
            lang_ids=[norm_language] * minibatch_size,
        )
        pred_text = [output["text"] for output in outputs]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

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

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Reload dataset for actual evaluation (reset streaming pointer)
    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
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
            all_results["references"], all_results["predictions"],
            all_results["audio_length_s"], all_results["transcription_time_s"],
            all_results["audio_filepath"]
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        (all_results["references"], all_results["predictions"], all_results["audio_length_s"],
         all_results["transcription_time_s"], all_results["audio_filepath"]) = zip(*filtered)
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

    if norm_language in OIWER_LANGUAGES:
        # Lattice-based, orthography-aware scoring (voi_oiwer applies its own
        # normalization internally).
        manifest = [
            {"text": ref, "pred_text": pred}
            for ref, pred in zip(all_results["references"], all_results["predictions"])
        ]
        wer, _ins, _del, _sub = score_oiwer(manifest, OIWER_LANGUAGES[norm_language])
    else:
        norm_refs = [data_utils.ml_normalizer(r, lang=norm_language) for r in all_results["references"]]
        norm_preds = [data_utils.ml_normalizer(p, lang=norm_language) for p in all_results["predictions"]]
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
        "--dataset",
        type=str,
        required=True,
        help="Dataset path, e.g. `hf-audio/open-asr-leaderboard-multilingual-datasets`.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name, e.g. `fleurs_de`. Omit for single-config dataset repos.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code used for decoding and normalization, e.g. `de`.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset.",
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
        help="Maximum number of tokens to generate.",
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
