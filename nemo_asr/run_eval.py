import argparse

import io
import os
import torch
import evaluate
import soundfile

from tqdm import tqdm
from normalizer import data_utils
import numpy as np

from nemo.collections.asr.models import ASRModel
import time


wer_metric = evaluate.load("wer")


def main(args):

    DATA_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")
    DATASET_NAME = args.dataset
    SPLIT_NAME = args.split

    CACHE_DIR = os.path.join(DATA_CACHE_DIR, DATASET_NAME, SPLIT_NAME)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        compute_dtype=torch.bfloat16
    else:
        device = torch.device("cpu")
        compute_dtype=torch.float32
        

    if args.model_id.endswith(".nemo"):
        asr_model = ASRModel.restore_from(args.model_id, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(args.model_id, map_location=device)  # type: ASRModel
    
    asr_model.to(compute_dtype)
    asr_model.eval()

    dataset = data_utils.load_data(args)

    def download_audio_files(batch):

        # download audio files and write the paths, transcriptions and durations to a manifest file
        audio_paths = []
        durations = []

        for id, sample in zip(batch["id"], batch["audio"]):

            # frist step added here to make ID and wav filenames unique
            # several datasets like earnings22 have a hierarchical structure
            # for eg. earnings22/test/4432298/281.wav, earnings22/test/4450488/281.wav
            # lhotse uses the filename (281.wav) here as unique ID to create and name cuts
            # ref: https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/collation.py#L186
            id = id.replace('/', '_').removesuffix('.wav')

            audio_path = os.path.join(CACHE_DIR, f"{id}.wav")

            if "array" in sample:
                audio_array = np.float32(sample["array"])
                sample_rate = 16000

            elif "bytes" in sample: # added to be compatible with latest datasets library (3.x.x) that produces byte stream
                with io.BytesIO(sample["bytes"]) as audio_file:
                    audio_array, sample_rate = soundfile.read(audio_file, dtype="float32")

            else:
                raise ValueError("Sample must have either 'array' or 'bytes' key")

            if not os.path.exists(audio_path):
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                soundfile.write(audio_path, audio_array, sample_rate)

            audio_paths.append(audio_path)
            durations.append(len(audio_array) / sample_rate)

        
        batch["references"] = batch["norm_text"]
        batch["audio_filepaths"] = audio_paths
        batch["durations"] = durations

        return batch


    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples !")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)
    if asr_model.cfg.decoding.strategy != "beam":
        asr_model.cfg.decoding.strategy = "greedy_batch"
        asr_model.change_decoding_strategy(asr_model.cfg.decoding)
    
    # prepraing the offline dataset
    dataset = dataset.map(download_audio_files, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    # Write manifest from daraset batch using json and keys audio_filepath, duration, text

    all_data = {
        "audio_filepaths": [],
        "durations": [],
        "references": [],
    }

    data_itr = iter(dataset)
    for data in tqdm(data_itr, desc="Downloading Samples"):
        for key in all_data:
            all_data[key].append(data[key])
    
    # Sort audio_filepaths and references based on durations values
    sorted_indices = sorted(range(len(all_data["durations"])), key=lambda k: all_data["durations"][k], reverse=True)
    all_data["audio_filepaths"] = [all_data["audio_filepaths"][i] for i in sorted_indices]
    all_data["references"] = [all_data["references"][i] for i in sorted_indices]
    all_data["durations"] = [all_data["durations"][i] for i in sorted_indices]
    
    
    total_time = 0
    for _ in range(2): # warmup once and calculate rtf
        if _ == 0:
            audio_files = all_data["audio_filepaths"][:args.batch_size * 4] # warmup with 4 batches
        else:
            audio_files = all_data["audio_filepaths"]
        start_time = time.time()
        with torch.cuda.amp.autocast(enabled=False, dtype=compute_dtype), torch.inference_mode(), torch.no_grad():
            if 'canary' in args.model_id:
                transcriptions = asr_model.transcribe(audio_files, batch_size=args.batch_size, verbose=False, pnc='no', num_workers=1)
            else:
                transcriptions = asr_model.transcribe(audio_files, batch_size=args.batch_size, verbose=False, num_workers=1)
        end_time = time.time()
        if _ == 1:
            total_time += end_time - start_time
    total_time = total_time

    # normalize transcriptions with English normalizer
    if isinstance(transcriptions, tuple) and len(transcriptions) == 2:
        transcriptions = transcriptions[0]
    predictions = [data_utils.normalizer(pred.text) for pred in transcriptions]

    avg_time = total_time / len(all_data["audio_filepaths"])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_data["references"],
        predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_data["durations"],
        transcription_time=[avg_time] * len(all_data["audio_filepaths"]),
    )

    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=all_data['references'], predictions=predictions)
    wer = round(100 * wer, 2)

    # transcription_time = sum(all_results["transcription_time"])
    audio_length = sum(all_data["durations"])
    rtfx = audio_length / total_time
    rtfx = round(rtfx, 2)

    print("RTFX:", rtfx)
    print("WER:", wer, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with NVIDIA NeMo.",
    )
    parser.add_argument(
        '--dataset_path', type=str, default='esb/datasets', help='Dataset path. By default, it is `esb/datasets`'
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
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args)
