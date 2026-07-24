import argparse
import os
import re
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoModelForMultimodalLM, AutoModelForCTC, AutoProcessor, MODEL_FOR_MULTIMODAL_LM_MAPPING, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING, MODEL_FOR_CTC_MAPPING, CompileConfig
import evaluate
from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs
from tqdm import tqdm
from datasets import load_dataset, Audio
import random
import numpy as np

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


def remove_brackets(text):
    """
    Remove parentheses from text, replacing them with spaces.

    Some models (e.g. Cohere ASR) output parentheses that would cause the
    normalizer to delete the enclosed text entirely, leading to false
    deletion errors in the predictions.
    """
    text = text.replace("(", " ").replace(")", " ")
    # replace spans of multiple spaces as a single space
    text = re.sub(r"\s+", " ", text)
    return text


def main(args):

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch_dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
    if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        cls_model = AutoModelForSpeechSeq2Seq
    elif type(config) in MODEL_FOR_MULTIMODAL_LM_MAPPING:
        cls_model = AutoModelForMultimodalLM
    elif type(config) in MODEL_FOR_CTC_MAPPING:
        cls_model = AutoModelForCTC
    else:
        raise ValueError(f"Model config of type {type(config)} not recognized in Transformers mappings.")
    is_ctc = cls_model == AutoModelForCTC

    model = cls_model.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
    )
    model.to(args.device)
    model.eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    has_transcription_processor = hasattr(processor, "apply_transcription_request")
    is_cohere = "cohere" in args.model_id.lower() and "transcribe" in args.model_id.lower()
    # Voxtral Realtime uses a simple processor call (no apply_transcription_request / prompt)
    is_voxtral_realtime = "voxtral" in args.model_id.lower() and "realtime" in args.model_id.lower()

    # Extract sampling rate from processor
    if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
        sampling_rate = processor.feature_extractor.sampling_rate
    elif hasattr(processor, "audio_processor") and processor.audio_processor is not None:
        sampling_rate = processor.audio_processor.sampling_rate
    else:
        sampling_rate = 16_000

    # Set generate arguments (only for auto-regressive models)
    if model.can_generate():
        gen_kwargs = {}
        if args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens

        # For multilingual models, set task to transcribe and pass language (None = auto-detect)
        if getattr(model.generation_config, "is_multilingual", False):
            gen_kwargs["task"] = "transcribe"
            if args.language is not None:
                gen_kwargs["language"] = args.language
    elif args.max_new_tokens:
        raise ValueError("`max_new_tokens` should only be set for auto-regressive models, but got a CTC model.")

    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split

    # Determine language for normalization: use --language if provided, otherwise extract from config_name
    if args.language is not None:
        norm_language = args.language
    else:
        try:
            norm_language = CONFIG_NAME.split("_", 1)[1]
        except IndexError:
            norm_language = "en"
        print(f"Language not specified, extracted '{norm_language}' from config_name '{CONFIG_NAME}'")

    if args.torch_compile is not None:
        if model.can_generate():
            gen_kwargs["compile_config"] = CompileConfig(mode=args.torch_compile, fullgraph=args.compile_fullgraph)
            model.generation_config.cache_implementation = "static"
        else:
            model = torch.compile(model, mode=args.torch_compile, fullgraph=args.compile_fullgraph)

        # Ensure warm-up runs when using torch.compile
        if args.warmup_steps is None or args.warmup_steps < 1:
            print("`--torch_compile` is enabled; forcing `--warmup_steps=10` to trigger compilation before timed runs.")
            args.warmup_steps = 10

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

    def benchmark(batch, min_new_tokens=None):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        sampling_rate = batch["audio"][0]["sampling_rate"]
        batch["audio_length_s"] = [len(audio) / sampling_rate for audio in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        # START TIMING
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # 1. Pre-Processing
        # Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile is not None:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        if is_cohere:
            # Cohere ASR requires an explicit language and does not use apply_transcription_request
            inputs = processor(
                audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                language=norm_language,
                punctuation=norm_language != "en",
            )
        elif is_voxtral_realtime:
            # Realtime model uses a plain processor call — no prompt, no apply_transcription_request
            inputs = processor(audios, return_tensors="pt")
        elif has_transcription_processor:
            if "voxtral" in args.model_id.lower():
                inputs = processor.apply_transcription_request(
                    language=args.language,  # None = auto-detect
                    audio=audios,
                    model_id=args.model_id,
                    sampling_rate=sampling_rate,
                    format=["wav"] * len(audios),
                )
            else:
                inputs = processor.apply_transcription_request(audios)
            prompt_len = inputs["input_ids"].shape[1]
        elif not model.can_generate():
            # CTC pre-processing: normalize to mean 0, std 1
            inputs = processor(
                audios,
                sampling_rate=sampling_rate,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            # Standard Whisper processing: pad audios to 30-seconds and convert to log-mel
            inputs = processor(
                audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
                device=args.device,
            )

        inputs = inputs.to(args.device, dtype=torch_dtype)

        # 2. Model Inference
        if args.torch_compile is not None:
            sdpa_backends = [SDPBackend.MATH]
        else:
            sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        with sdpa_kernel(sdpa_backends):
            if model.can_generate():
                pred_ids = model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)
            else:
                # Single forward pass for CTC
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = logits.argmax(-1)

        # 3. Post-processing
        # Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # Convert token ids to text transcription
        if is_cohere:
            audio_chunk_index = inputs.get("audio_chunk_index")
            pred_text = processor.decode(
                pred_ids,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=norm_language,
            )
            pred_text = [remove_brackets(t) for t in pred_text]
        elif is_voxtral_realtime:
            # No prompt tokens to strip — decode directly
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        elif has_transcription_processor:
            pred_text = processor.batch_decode(pred_ids[:, prompt_len:], skip_special_tokens=True)
        elif is_ctc:
            # don't use skip_special_tokens as it collapses double letters
            pred_text = processor.batch_decode(pred_ids)
        else:
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["text"]  # raw; normalization applied at scoring time

        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(
            benchmark, batch_size=args.batch_size, batched=True,
            fn_kwargs={"min_new_tokens": args.max_new_tokens}
        ))
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
        all_results["references"], all_results["predictions"], all_results["audio_length_s"], all_results["transcription_time_s"], all_results["audio_filepath"] = zip(*filtered)
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
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

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
        required=True,
        help="Model identifier. Should be loadable with Transformers",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use (e.g. 'refs/pr/11' for a PR branch). Defaults to the main branch.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. E.g. 'hf-audio/open-asr-leaderboard-multilingual-datasets'",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name for the dataset. E.g. 'fleurs_de' for German FLEURS.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code, e.g. 'de' for German. If not set, the model will auto-detect the language.",
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
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of samples to go through each batch.",
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
        "--torch_compile",
        type=str,
        default=None,
        help="Mode for torch compiling model forward pass. Can be either 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs'.",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Whether to do full graph compilation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="The dtype to use for model loading and inference. E.g. 'bfloat16', 'float16', 'float32'.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention implementation to use for model loading (e.g. 'sdpa', 'eager', 'flash_attention_2').",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
