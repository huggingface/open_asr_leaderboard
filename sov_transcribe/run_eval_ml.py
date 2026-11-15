import argparse
import os
import re
import time

import evaluate
from datasets import load_dataset, Audio
from huggingface_hub import get_safetensors_metadata
from tqdm import tqdm
from vllm import LLM, SamplingParams

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs

wer_metric = evaluate.load("wer")

SAMPLING_RATE = 16000

# The model card's instruction for the Open ASR multilingual sets: one prompt for every language (the model picks
# the language from the audio; no language hint is given).
PROMPT = (
    "Transcribe the audio verbatim in the language's standard writing system, using punctuation and "
    "capitalization. Output only the transcript."
)
# The languages of the model card (English is evaluated with run_eval.py).
SUPPORTED_LANGUAGES = {"de", "es", "fr", "it", "nl", "pt"}


def main(args):
    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split

    # Determine language for normalization: use --language if provided, otherwise
    # extract from config_name (e.g. "fleurs_fr") or, for single-config repos, the repo name
    if args.language:
        LANGUAGE = args.language
    else:
        source = CONFIG_NAME if CONFIG_NAME else os.path.basename(args.dataset)
        lang_match = re.search(r"_([a-z]{2})(?:_test)?$", source)
        LANGUAGE = lang_match.group(1) if lang_match else "en"
    if LANGUAGE not in SUPPORTED_LANGUAGES:
        raise ValueError(f"SOV-Transcribe-1 does not support '{LANGUAGE}' (supported: {sorted(SUPPORTED_LANGUAGES)}).")
    print(f"Language: {LANGUAGE}, prompt: {PROMPT!r}")

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
        [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "audio"}]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    def load(max_samples):
        dataset = load_dataset(
            args.dataset,
            CONFIG_NAME,
            split=SPLIT_NAME,
            streaming=args.streaming,
            token=True,
        )
        # Resample to 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        if max_samples is not None and max_samples > 0:
            if args.streaming:
                dataset = dataset.take(max_samples)
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / SAMPLING_RATE for audio in audios]
        minibatch_size = len(audios)

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
        batch["references"] = batch["text"]  # raw; normalization applied at scoring time
        return batch

    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Running {args.warmup_steps} warmup steps...")
        warmup_dataset = load(args.warmup_steps * args.batch_size)
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"]))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
    dataset = load(args.max_eval_samples)
    dataset = dataset.map(
        benchmark, batch_size=None, batched=True, remove_columns=["audio"],
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
        CONFIG_NAME or "",
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
        help="Model identifier. Should be loadable with vLLM and the SOV-Transcribe vLLM plugin.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (commit hash or tag) on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'hf-audio/open-asr-leaderboard-multilingual-datasets'`",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name for the dataset. *E.g.* `'fleurs_fr'` for French FLEURS. "
             "Omit for single-config dataset repos.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'fr'). If not provided, extracted from config_name.",
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
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (same as run_eval.py).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="vLLM context length (same as run_eval.py).",
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
