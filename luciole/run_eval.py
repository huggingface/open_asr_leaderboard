import argparse
import os
import time

import evaluate
from huggingface_hub import get_safetensors_metadata
from tqdm import tqdm
from vllm import LLM, SamplingParams

from normalizer import data_utils

wer_metric = evaluate.load("wer")

# Audio placeholder of the NeMo SpeechLM vLLM plugin (nemo.collections.speechlm2.vllm.salm),
# registered through the `vllm.general_plugins` entry point of nemo-toolkit.
AUDIO_LOCATOR_TAG = "<|audio|>"
SAMPLING_RATE = 16000


def main(args):
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
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": f"{args.prompt} {AUDIO_LOCATOR_TAG}"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / SAMPLING_RATE for audio in audios]
        minibatch_size = len(audios)
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

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
        batch["references"] = batch["original_text"]  # raw; normalization applied at scoring time
        return batch

    if args.warmup_steps is not None:
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
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
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
        help="Model identifier. Should be loadable with vLLM and the NeMo SpeechLM plugin.",
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
        default="Transcribe in English",
        help="ASR instruction; the audio locator tag is appended to it.",
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
