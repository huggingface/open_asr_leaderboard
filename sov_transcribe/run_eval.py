import argparse
import os
import time

import evaluate
from huggingface_hub import get_safetensors_metadata
from tqdm import tqdm
from vllm import LLM, SamplingParams

from normalizer import data_utils

wer_metric = evaluate.load("wer")

SAMPLING_RATE = 16000

# Transcription-mode instructions, verbatim from the model card's evaluation protocol. Apart from the mode
# (verbatim vs. clean, and the script for Voice Arena's romanized references), the instruction is the same for
# every dataset. The submit scripts pick the style per dataset.
PROMPTS = {
    # Verbatim references: AMI, Earnings22, GigaSpeech, URGENT24, VoxPopuli, the conversational private sets.
    "verbatim": (
        "Transcribe the audio verbatim, including fillers and repetitions. Use punctuation and capitalization. "
        "Output only the transcript."
    ),
    # Voice Arena (Monsoon English): verbatim, with references in Latin script.
    "verbatim_latin": (
        "Transcribe the audio verbatim, including fillers and repetitions. Use punctuation and capitalization. "
        "Output only the transcript. Use Latin script only."
    ),
    # Non-verbatim references (SPGISpeech, the scripted private sets): no fillers or accidental repetitions.
    "clean": (
        "Transcribe the audio, omitting fillers and accidental repetitions while preserving the remaining words. "
        "Use punctuation and capitalization. Output only the transcript."
    ),
}


def main(args):
    prompt_text = PROMPTS[args.prompt_style]
    print(f"Prompt style: {args.prompt_style}, prompt: {prompt_text!r}")

    # The model is served by the vLLM plugin shipped in the model repo (vllm_plugin/, installed in the Space's
    # Dockerfile), registered through the `vllm.general_plugins` entry point.
    llm = LLM(
        model=args.model_id,
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
        # The warmup samples are the dataset's first ones: don't reuse their preprocessed audio
        mm_processor_cache_gb=0,
        compilation_config={"max_cudagraph_capture_size": max(512, args.batch_size)},
    )
    # vLLM does not expose the module tree, so count parameters from the checkpoint metadata
    num_params = sum(get_safetensors_metadata(args.model_id, revision=args.revision).parameter_count.values())
    print(f"Model size: {num_params / 1e9:.2f}B parameters")
    # Greedy decoding, identical for every dataset (the model card's sampling parameters)
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_tokens=args.max_new_tokens
    )

    # Same prompt as the plugin builds for /v1/audio/transcriptions: the instruction and the audio in one user
    # turn, with thinking disabled. The audio item renders as the model's `<|audio|>` placeholder.
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "audio"}]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / SAMPLING_RATE for audio in audios]
        minibatch_size = len(audios)
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        # START TIMING
        start_time = time.time()

        # INFERENCE: vLLM schedules all requests with continuous batching
        outputs = llm.generate(
            [
                {"prompt_token_ids": prompt_token_ids, "multi_modal_data": {"audio": (audio, SAMPLING_RATE)}}
                for audio in audios
            ],
            sampling_params,
            use_tqdm=False,
        )
        pred_text = [output.outputs[0].text.strip() for output in outputs]

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark, batch_size=None, batched=True, remove_columns=["audio"],
    )

    is_chunked = data_utils.is_chunked_dataset(args.dataset_path)

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }
    if is_chunked:
        all_results.update({key: [] for key in data_utils.CHUNK_METADATA_KEYS})
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
        extra_fields={key: all_results[key] for key in data_utils.CHUNK_METADATA_KEYS}
        if is_chunked
        else None,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

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
        help="Model identifier. Should be loadable with vLLM and the SOV-Transcribe vLLM plugin.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (commit hash or tag) on the Hugging Face Hub.",
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
        default=0,
        help="Unused: vLLM runs on the visible GPU (set CUDA_VISIBLE_DEVICES to choose it). Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="vLLM's max_num_seqs (requests in flight); each warmup step sends this many samples.",
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
        "--prompt_style",
        type=str,
        choices=sorted(PROMPTS),
        default="verbatim",
        help="Transcription mode (see PROMPTS): verbatim, verbatim_latin (Voice Arena) or clean (non-verbatim references).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (well above any short-form transcript).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="vLLM context length: room for about 8 minutes of audio (375 tokens per 30 s window) and its transcript.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM may use.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
