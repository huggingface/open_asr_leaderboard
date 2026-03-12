import argparse
import os
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
import numpy as np
import random

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

def main(args):
    print("=" * 20)
    print("Dataset:", args.dataset)
    print("Split:", args.split)
    print("=" * 20)

    start_time_overall = time.time()

    # set seed due to randomness in acoustic tokenizer sampling
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model
    model_id = args.model_id
    processor = AutoProcessor.from_pretrained(model_id)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id, 
        dtype=torch.bfloat16,
        attn_implementation={
            "acoustic_tokenizer_encoder_config": "eager",
            "semantic_tokenizer_encoder_config": "eager",
            "text_config": "sdpa",
        }
    ).to(args.device)
    print(f"Model loaded on {model.device} with dtype {model.dtype}")

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        start_time = time.time()

        # Prepare batch inputs
        batch_inputs = processor.apply_transcription_request(audios).to(model.device, model.dtype)

        # Model inference
        with torch.no_grad():
            output_ids = model.generate(**batch_inputs, max_new_tokens=args.max_new_tokens)

        # Slice off the input prompt for all samples
        prompt_len = batch_inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        inference_time = time.time() - start_time

        # Decode as a batch with fallback to individual decoding on error
        try:
            decoding_start_time = time.time()
            pred_text = processor.decode(generated_ids, return_format="transcription_only")
        except Exception as e:
            print(f"Batch decoding failed with error: {e}. Falling back to individual sample decoding.")
            decoding_start_time = time.time()
            pred_text = []
            for i, sample_ids in enumerate(generated_ids):
                try:
                    decoded = processor.decode(sample_ids.unsqueeze(0), return_format="transcription_only")
                    pred_text.append(decoded[0] if isinstance(decoded, list) else decoded)
                except Exception as sample_error:
                    print(f"Sample {i} decoding failed with error: {sample_error}. Setting to empty transcript.")
                    pred_text.append("")
        decoding_time = time.time() - decoding_start_time
        
        runtime = inference_time + decoding_time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    print("Loading dataset...")
    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=processor.feature_extractor.sampling_rate)
    print("Dataset loaded.")

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
    print("Inference completed on all samples.")

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)

    total_runtime = time.time() - start_time_overall
    print(f"Total evaluation runtime: {total_runtime / 60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted", help="Dataset path. By default, it is `hf-audio/esb-datasets-test-only-sorted`")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset.")
    parser.add_argument("--device", type=int, default=-1, help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to go through each streamed batch.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Number of samples to be evaluated.")
    parser.add_argument("--max_new_tokens", type=int, default=900, help="Maximum number of tokens to generate. Default is 900 (2 minutes of audio at 24000 kHz with 3200 compression). Maximum allowed is 32768 (60 minutes) as this was the maximum context size used during training.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Choose whether to download the entire dataset or stream it during the evaluation.")
    args = parser.parse_args()
    parser.set_defaults(streaming=False)
    main(args)
