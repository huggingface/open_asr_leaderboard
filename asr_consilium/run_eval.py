import argparse
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from normalizer import data_utils
from normalizer.data_utils import normalize
import time
import json
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset
from asr_consilium import inference, AVAILABLE_MODELS


wer_metric = evaluate.load("wer")


def store_test_dataset_as_files_unique(dataset, out_dir, name='test'):
    os.makedirs(out_dir, exist_ok=True)
    output_jsonl_file = os.path.join(out_dir, "markdown.jsonl")
    if os.path.isfile(output_jsonl_file):
        print("Dataset already created!")
        return output_jsonl_file
    out = open(output_jsonl_file, 'w', encoding='utf-8')
    print(dataset)
    print("Dataset length: {}".format(len(dataset[name])))
    for i in tqdm(range(len(dataset[name]))):
        # print(dataset[name][i])
        if dataset[name][i]["audio"]["path"] is None:
            orig_name = '{}.wav'.format(i)
        else:
            part = dataset[name][i]["audio"]["path"][:-4]
            part = part.replace(":", "")
            orig_name = os.path.basename(part + '_{}.wav'.format(i))
        audio = dataset[name][i]["audio"]["array"]
        sr = dataset[name][i]["audio"]["sampling_rate"]
        # print(out_dir, orig_name, os.path.join(os.path.abspath(out_dir), orig_name))
        sf.write(os.path.join(out_dir, orig_name), audio, sr, 'FLOAT')
        res = {
            'audio': orig_name,
            'text': dataset[name][i]['text'],
            'duration': len(audio) / sr,
        }
        out.write(json.dumps(res, ensure_ascii=False) + '\n')
    out.close()
    return output_jsonl_file


def main(args):
    cache_dir = os.path.dirname(os.path.abspath(__file__)) + '/cache/'
    os.makedirs(cache_dir, exist_ok=True)

    dt_path = args.dataset_path
    dt_name = args.dataset
    dt_type = args.split
    batch_size = int(args.batch_size)
    dataset = load_dataset(
        dt_path,
        dt_name,
        cache_dir=cache_dir,
    )
    dataset_folder = "open-asr-leaderboard-{}-{}".format(dt_name, dt_type)

    jsonl_dataset = store_test_dataset_as_files_unique(
        dataset,
        out_dir=cache_dir + dataset_folder,
        name=dt_type,
    )

    out_file = cache_dir + 'results_{}_{}.jsonl'.format(args.dataset, args.split)

    # Start timing
    start_time = time.time()

    inference(
        jsonl_file=jsonl_dataset,
        out_file=out_file,
        batch_size=batch_size,
        model_list=None,
        weights=None,
        language='en',
        normalize=True,
        char_level=False,
        skip_existed=True,
    )

    # End timing
    runtime = time.time() - start_time

    lines = open(jsonl_dataset, 'r', encoding="utf-8").readlines()
    items_orig = [json.loads(line) for line in lines]

    lines = open(out_file, 'r', encoding="utf-8").readlines()
    items_pred = [json.loads(line) for line in lines]

    single_entry_time = runtime / len(items_orig)

    full_data = {}
    for item_orig in items_orig:
        item = normalize(item_orig)
        if item['audio'] not in full_data:
            full_data[item['audio']] = {}
        full_data[item['audio']]['references'] = item['norm_text']
        full_data[item['audio']]['audio_length_s'] = item['duration']
        full_data[item['audio']]['transcription_time_s'] = single_entry_time

    for item in items_pred:
        full_data[item['audio']]['predictions'] = item['text']

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    keys = all_results.keys()
    for item, data in full_data.items():
        for key in keys:
            all_results[key].append(data[key])

    # Normalized datasets sometimes contains empty strings (like AMI)
    # So WER gives error: ValueError: one or more references are empty strings

    filtered_refs = []
    filtered_preds = []

    # Filter empty values
    for ref, pred in zip(all_results["references"], all_results["predictions"]):
        if ref and ref.strip():
            filtered_refs.append(ref)
            filtered_preds.append(pred)
        else:
            if len(ref.strip()) == 0 and len(pred.strip()) == 0:
                # Both are empty - so make it correct
                filtered_refs.append('1')
                filtered_preds.append('1')
            else:
                # Prediction is not empty - make it incorrect
                filtered_refs.append('1')
                filtered_preds.append(pred)

    all_results["references"] = filtered_refs
    all_results["predictions"] = filtered_preds

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
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
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
        help="Model identifier. Should be loadable with 🤗 Transformers",
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
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to go through each streamed batch.",
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
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)