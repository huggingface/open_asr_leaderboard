import os
import sys
import glob
import json

import evaluate


def read_manifest(manifest_path: str):
    data = []
    with open(manifest_path, "r", encoding='utf-8') as f:
        for line in f:
            if len(line) > 0:
                datum = json.loads(line)
                data.append(datum)
    return data


def write_manifest(
    references: list, transcriptions: list, model_id: str, dataset_path: str, dataset_name: str, split: str
):
    model_id = model_id.replace("/", "-")
    dataset_path = dataset_path.replace("/", "-")
    dataset_name = dataset_name.replace("/", "-")

    if len(references) != len(transcriptions):
        raise ValueError(
            f"The number of samples in `ground_truths` ({len(references)}) "
            f"must match `transcriptions` ({len(transcriptions)})."
        )

    basedir = './results/'
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    manifest_path = os.path.join(basedir, f"MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}.jsonl")

    with open(manifest_path, "w", encoding='utf-8') as f:
        for idx, (text, transcript) in enumerate(zip(references, transcriptions)):
            datum = {
                "audio_filepath": f"sample_{idx}",  # dummy value for Speech Data Processor
                "duration": 0.0,  # dummy value for Speech Data Processor
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path


def score_results(directory: str = None, model_id: str = None):
    if directory is None:
        # try to parse cmd line args
        args = sys.argv[1:]
        if len(args) == 0:
            raise ValueError("Please specify a directory containing result files.")

        directory = args[0]

        # see if model_id is specified
        if len(args) > 1 and args[1] != "":
            model_id = args[1]

    # Strip trailing slash
    if directory.endswith(os.pathsep):
        directory = directory[:-1]

    result_files = list(glob.glob(f"{directory}/**/*.jsonl", recursive=True))
    result_files = list(sorted(result_files))

    # Filter files belonging to a specific model id
    if model_id is not None:
        model_id = model_id.replace("/", "-")
        result_files = [fp for fp in result_files if model_id in fp]

    if len(result_files) == 0:
        raise ValueError(f"No result files found in {directory}")

    def parse_filepath(fp: str):
        model_index = fp.find("MODEL_")
        fp = fp[model_index:]
        ds_index = fp.find("DATASET_")
        model_id = fp[:ds_index].replace("MODEL_", "").rstrip("_")
        author_index = model_id.find("-")
        model_id = model_id[:author_index] + "/" + model_id[author_index + 1 :]

        ds_fp = fp[ds_index:]
        ds_fp = ds_fp.replace("DATASET_", "").rstrip(".jsonl")
        parts = ds_fp.split("_")
        dataset_path = parts[0]
        dataset_name = parts[1]
        split = parts[2]
        return model_id, dataset_path, dataset_name, split

    # Results per dataset
    results = {}
    wer_metric = evaluate.load("wer")

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_path, dataset_name, split = parse_filepath(result_file)

        references = [datum["text"] for datum in manifest]
        predictions = [datum["pred_text"] for datum in manifest]

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        result_key = f"{model_id_of_file} | {dataset_path}_{dataset_name}_{split}"
        results[result_key] = wer

    print("*" * 80)
    print("Results per dataset:")
    print("*" * 80)

    for k, v in results.items():
        print(f"{k}: WER = {v:0.2f} %")

    composite_wer = sum(results.values()) / len(results)
    print()
    print("*" * 80)
    print(f"Composite WER: {composite_wer:0.2f} %")

    return composite_wer
