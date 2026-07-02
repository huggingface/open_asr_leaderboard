import argparse
import os
import random
import re

import evaluate
import numpy as np
import torch
from normalizer import data_utils
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from transformers import (
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_MULTIMODAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    AutoConfig,
    AutoModelForCTC,
    AutoModelForMultimodalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    CompileConfig,
)


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def remove_brackets(text):
    """
    Remove parentheses from text, replacing them with spaces.

    Some models (e.g. Cohere ASR) output parentheses that would cause the
    normalizer to delete the enclosed text entirely, leading to false
    deletion errors in the predictions.
    """
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def main(args):
    # Set seed due to randomness in some models (e.g. VibeVoice's acoustic tokenizer sampling)
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

    if "vibevoice" in args.model_id.lower():
        model = cls_model.from_pretrained(
            args.model_id,
            dtype=torch_dtype,
            attn_implementation={
                "acoustic_tokenizer_encoder_config": "eager",
                "semantic_tokenizer_encoder_config": "eager",
                "text_config": "sdpa",
            },
        )
    else:
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
    is_qwen3_asr = "qwen3-asr" in args.model_id.lower()

    # Optional prompt for audio language models, newer models should use `apply_transcription_request`
    text = None
    if "granite-speech-3.3" in args.model_id.lower():
        # create text prompt
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|>can you transcribe the speech into a written format?",
            },
        ]

        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Extract sampling rate
    if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
        sampling_rate = processor.feature_extractor.sampling_rate
    elif hasattr(processor, "audio_processor") and processor.audio_processor is not None:
        sampling_rate = processor.audio_processor.sampling_rate
    else:
        sampling_rate = 16_000

    # Set generate arguments
    if model.can_generate():
        gen_kwargs = {}
        if args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens
        if getattr(model.generation_config, "is_multilingual", False):
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"
        # Clear deprecated Whisper generation config fields to suppress warnings
        if hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = None
        if hasattr(model.generation_config, "suppress_tokens"):
            model.generation_config.suppress_tokens = []
        if hasattr(model.generation_config, "begin_suppress_tokens"):
            model.generation_config.begin_suppress_tokens = []
        if "granite-speech-3.3" in args.model_id.lower():
            gen_kwargs["repetition_penalty"] = 1.0
    elif args.max_new_tokens:
        raise ValueError("`max_new_tokens` should only be set for auto-regressive models, but got a CTC model.")

    if args.torch_compile is not None:
        if model.can_generate():
            gen_kwargs["compile_config"] = CompileConfig(mode=args.torch_compile, fullgraph=args.compile_fullgraph)
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = "static"
        else:
            model = torch.compile(model, mode=args.torch_compile, fullgraph=args.compile_fullgraph)

        # Ensure warm-up runs when using torch.compile
        if args.warmup_steps is None or args.warmup_steps < 1:
            print(
                "`--torch_compile` is enabled; forcing `--warmup_steps=10` to trigger compilation before timed runs."
            )
            args.warmup_steps = 10

    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        sampling_rate = batch["audio"][0]["sampling_rate"]
        batch["audio_length_s"] = [len(audio) / sampling_rate for audio in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)
        if text is not None:
            texts = [text] * minibatch_size
        else:
            texts = None

        # START TIMING
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # 1. Pre-Processing
        # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile is not None:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        if is_cohere:
            inputs = processor(
                audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                language="en",
                punctuation=False,
            )
        elif is_voxtral_realtime:
            # Realtime model uses a plain processor call — no prompt, no apply_transcription_request
            inputs = processor(audios, return_tensors="pt")
        elif has_transcription_processor:
            if "voxtral" in args.model_id.lower():
                inputs = processor.apply_transcription_request(
                    language="en",  # English for benchmark consistency
                    audio=audios,
                    model_id=args.model_id,
                    sampling_rate=sampling_rate,
                    format=["wav"] * len(audios),
                )
            elif is_qwen3_asr:
                inputs = processor.apply_transcription_request(
                    audios, language="en"
                )  # English for benchmark consistency
            else:
                inputs = processor.apply_transcription_request(audios)
            prompt_len = inputs["input_ids"].shape[1]
        elif texts is not None:
            inputs = processor(
                texts,
                audios,
                device=args.device,  # Computation device; returned tensors are put on CPU
                return_tensors="pt",
            ).to(args.device)
            prompt_len = inputs["input_ids"].shape[1]
        elif not model.can_generate():  # or len(audios[0]) > processor.feature_extractor.n_samples:
            # 1.2 Either CTC pre-processing (normalize to mean 0, std 1), or long-form Whisper processing
            inputs = processor(
                audios,
                sampling_rate=sampling_rate,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            # 1.3 Standard Whisper processing: pad audios to 30-seconds and converted to log-mel
            if args.longform:
                inputs = processor(
                    audios,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    return_attention_mask=True,
                )
            else:
                inputs = processor(
                    audios,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding="longest",
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
                # 2.1 Auto-regressive generation for LM-based models
                is_moonshine = "moonshine" in args.model_id.lower()
                if is_moonshine:
                    # Moonshine needs a per-sample token limit based on audio duration to
                    # prevent hallucinations on long/variable-length clips (e.g. AMI).
                    # Compute per-sample limits (6.5 tokens/sec + 2 for <sot>/<eot>),
                    # generate up to the batch maximum, then mask out tokens beyond each
                    # sample's individual limit by replacing them with the EOT token.
                    token_limits = [int(len(clip) * 6.5 // 16000 + 2) for clip in audios]
                    per_sample_limits = torch.tensor(token_limits).reshape(-1, 1).to(args.device)
                    moonshine_gen_kwargs = {**gen_kwargs, "max_new_tokens": per_sample_limits.max().item()}
                    pred_ids = model.generate(**inputs, **moonshine_gen_kwargs)
                    output_mask = (
                        torch.arange(pred_ids.shape[-1], device=args.device).unsqueeze(0).expand(pred_ids.shape[0], -1)
                        > per_sample_limits
                    )
                    pred_ids = pred_ids.masked_fill(output_mask, model.config.eos_token_id)
                elif args.longform:
                    pred_ids = model.generate(**inputs, **gen_kwargs, return_timestamps=True)
                else:
                    pred_ids = model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)
            else:
                # 2.2. Single forward pass for CTC
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = logits.argmax(-1)

        # 3. Post-processing
        # 3.1 Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # 3.2 Convert token ids to text transcription
        if is_cohere:
            audio_chunk_index = inputs.get("audio_chunk_index")
            pred_text = processor.decode(
                pred_ids,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language="en",
            )
            pred_text = [remove_brackets(t) for t in pred_text]
        elif "vibevoice" in args.model_id.lower():
            # VibeVoice: strip the input prompt tokens then use the model's own decode API
            generated_ids = pred_ids[:, prompt_len:]
            try:
                pred_text = processor.decode(generated_ids, return_format="transcription_only")
            except Exception as e:
                print(f"Batch decoding failed with error: {e}. Falling back to individual sample decoding.")
                pred_text = []
                for i, sample_ids in enumerate(generated_ids):
                    try:
                        decoded = processor.decode(sample_ids.unsqueeze(0), return_format="transcription_only")
                        pred_text.append(decoded[0] if isinstance(decoded, list) else decoded)
                    except Exception as sample_error:
                        print(f"Sample {i} decoding failed with error: {sample_error}. Setting to empty transcript.")
                        pred_text.append("")
        elif is_voxtral_realtime:
            # No prompt tokens to strip — decode directly
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        elif is_qwen3_asr:
            # Use Qwen3 ASR's structured decode to extract transcription text
            pred_text = processor.decode(pred_ids[:, prompt_len:], return_format="transcription_only")
        elif has_transcription_processor or texts is not None:
            # Strip input prompt tokens
            pred_text = processor.decode(pred_ids[:, prompt_len:], skip_special_tokens=True)
        elif is_ctc:
            # don't use skip_special_tokens as it collapses double letters
            pred_text = processor.batch_decode(pred_ids)
        else:
            pred_text = processor.decode(pred_ids, skip_special_tokens=True)

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(
            warmup_dataset.map(
                benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}
            )
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
    dataset = data_utils.prepare_data(dataset, sampling_rate=sampling_rate)

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
        "audio_filepath": [],
    }
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
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_refs = [data_utils.normalizer(r) for r in all_results["references"]]
    norm_preds = [data_utils.normalizer(p) for p in all_results["predictions"]]
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
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
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--longform",
        action="store_true",
        help="Whether to use longform mode.",
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
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use (e.g. 'refs/pr/11' for a PR branch). Defaults to the main branch.",
    )
    args = parser.parse_args()

    print("*" * 100)
    print(f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
