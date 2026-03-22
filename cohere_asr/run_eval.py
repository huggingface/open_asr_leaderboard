import argparse
import os
import re
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoConfig
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")


def remove_brackets(text):
    """
    Cohere ASR model sometimes outputs braces but the HF normalizers remove all text between braces
    during normalization implying deletion errors in the predictions.

    This function removes the braces from predictions before normalization. This is only 
    applied to predictions and not references.
    """
    text = text.replace("(", " ").replace(")", " ")
    # replace spans of multiple spaces as a single space
    text = re.sub(r'\s+', ' ', text)
    return text

@torch.inference_mode()
def main(args):
    args.model_id = os.path.normpath(args.model_id)

    trust_remote_code = True
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    model = load_model(args.model_id, device, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=trust_remote_code)

    def build_records(dataset_iter, desc):
        records = []
        for sample in tqdm(dataset_iter, desc=desc):
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            records.append(
                {
                    "audio_array": audio_array,
                    "sampling_rate": sampling_rate,
                    "reference": sample["original_text"],
                    "reference_norm_en": sample["norm_text"],
                    "audio_length_s": len(audio_array) / sampling_rate,
                }
            )
        return records

    def run_batched_inference(records, desc, collect_results):
        audio_lengths = []
        transcription_times = []
        predictions = []
        references = []

        for start_idx in tqdm(range(0, len(records), args.batch_size), desc=desc):
            batch_records = records[start_idx:start_idx + args.batch_size]
            audios = [record["audio_array"] for record in batch_records]
            sample_rates = [record["sampling_rate"] for record in batch_records]

            start_time = time.time()
            # following canary-1b and canary-1b-flash, we set punctuation=False for English and for other languages, we set punctuation=True 
            batch_predictions = model.transcribe(
                processor=processor,
                audio_arrays=audios,
                sample_rates=sample_rates,
                language=args.language,
                punctuation=args.language != "en",
                batch_size=args.batch_size,
                compile=not args.no_torch_compile,
            )
            runtime = time.time() - start_time
            per_sample_runtime = runtime / len(batch_records)

            if not collect_results:
                continue

            audio_lengths.extend(record["audio_length_s"] for record in batch_records)
            transcription_times.extend([per_sample_runtime] * len(batch_records))

            normalizer = data_utils.normalizer if args.language == 'en' else data_utils.ml_normalizer
            predictions.extend(normalizer(remove_brackets(pred)) for pred in batch_predictions)
            if args.language == 'en':
                references.extend(record["reference_norm_en"] for record in batch_records)
            else:
                references.extend(normalizer(record["reference"]) for record in batch_records)

        return audio_lengths, transcription_times, predictions, references

    if args.warmup_steps is not None:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_records = build_records(iter(warmup_dataset), desc="Preparing warm-up samples...")
        run_batched_inference(warmup_records, desc="Warming up...", collect_results=False)

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    records = build_records(iter(dataset), desc="Preparing samples...")
    records.sort(key=lambda record: record["audio_length_s"], reverse=True)
    (
        audio_lengths,
        transcription_times,
        predictions,
        references,
    ) = run_batched_inference(records, desc="Samples...", collect_results=True)

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        references,
        predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=audio_lengths,
        transcription_time=transcription_times,
        basedir=args.basedir,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=references, predictions=predictions
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(audio_lengths) / sum(transcription_times), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)

def load_model(model_id, device, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Loading model: {model_id}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the dataset. *E.g.* `'en'` for English, or `'fr'` for French.",
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
        default=256,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--no_torch_compile",
        action="store_true",
        help="Whether to NOT use torch compile.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    parser.add_argument(
        "--basedir",
        type=str,
        default="./results/",
        help="Base directory to save the manifest file.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
