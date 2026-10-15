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

# Audio placeholder of the NeMo SpeechLM vLLM plugin (nemo.collections.speechlm2.vllm.salm),
# registered through the `vllm.general_plugins` entry point of nemo-toolkit.
AUDIO_LOCATOR_TAG = "<|audio|>"
SAMPLING_RATE = 16000

# ASR instruction per language, used unless --prompt is given: the same training prompt in every
# language ("word-for-word transcription of the spoken content") with the target language added.
# It keeps the model from answering questions spoken in the audio.
PROMPTS = {
    "en": "Make a word-for-word transcription in English of the spoken content.",
    "fr": "Fais une transcription mot pour mot en français du contenu parlé.",
    "de": "Erstelle eine wortwörtliche Transkription des gesprochenen Inhalts auf Deutsch.",
    "es": "Haz una transcripción palabra por palabra en español del contenido hablado.",
    "it": "Fai una trascrizione parola per parola in italiano del contenuto parlato.",
    "nl": "Maak een woord-voor-woord transcriptie in het Nederlands van de gesproken inhoud.",
    "pt": "Faça uma transcrição palavra por palavra em português do conteúdo falado.",
}


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
    prompt_text = args.prompt or PROMPTS.get(LANGUAGE)
    if prompt_text is None:
        raise ValueError(f"No default prompt for language '{LANGUAGE}', pass --prompt.")
    print(f"Language: {LANGUAGE}, prompt: {prompt_text!r}")

    llm = LLM(
        model=args.model_id,
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
    )
    # vLLM does not expose the module tree, so count parameters from the checkpoint metadata
    num_params = sum(get_safetensors_metadata(args.model_id, revision=args.revision).parameter_count.values())
    print(f"Model size: {num_params / 1e9:.2f}B parameters")
    # Greedy decoding, identical for every dataset
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    # Same chat template as the vLLM OpenAI server applies to a "<prompt> <|audio|>" user turn
    messages = [{"role": "user", "content": f"{prompt_text} {AUDIO_LOCATOR_TAG}"}]
    if args.system_prompt:
        messages.insert(0, {"role": "system", "content": args.system_prompt})
    prompt = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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

        # INFERENCE: vLLM schedules the whole minibatch with continuous batching
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"audio": (audio, SAMPLING_RATE)}} for audio in audios],
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
        CONFIG_NAME or "",
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    if LANGUAGE == "en":
        norm_refs = [data_utils.normalizer(r) for r in all_results["references"]]
        norm_preds = [data_utils.normalizer(p) for p in all_results["predictions"]]
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
        required=True,
        help="Model identifier. Should be loadable with vLLM and the NeMo SpeechLM plugin.",
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
        help="Number of samples sent to vLLM per call, also used as vLLM's max_num_seqs.",
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
        "--prompt",
        type=str,
        default=None,
        help="ASR instruction; the audio locator tag is appended to it. Defaults to a per-language prompt.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system message; by default the chat template's own one is used.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="vLLM context length (the model's context window is 16,384 tokens).",
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
