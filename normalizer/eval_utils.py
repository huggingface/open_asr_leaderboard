import os
import glob
import json

import evaluate
from collections import defaultdict

def read_manifest(manifest_path: str):
    """
    Reads a manifest file (jsonl format) and returns a list of dictionaries containing samples.
    """
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
    """
    Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        references: Ground truth reference texts.
        transcriptions: Model predicted transcriptions.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.

    Returns:
        Path to the manifest file.
    """
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


def score_results(directory: str, model_id: str = None):
    """
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.

    Returns:
        Composite score over all evaluated datasets and a dictionary of all results.
    """

    # Strip trailing slash
    if directory.endswith(os.pathsep):
        directory = directory[:-1]

    # Find all result files in the directory
    result_files = list(glob.glob(f"{directory}/**/*.jsonl", recursive=True))
    result_files = list(sorted(result_files))

    # Filter files belonging to a specific model id
    if model_id is not None and model_id != "":
        print("Filtering models by id:", model_id)
        model_id = model_id.replace("/", "-")
        result_files = [fp for fp in result_files if model_id in fp]

    # Check if any result files were found
    if len(result_files) == 0:
        raise ValueError(f"No result files found in {directory}")

    # Utility function to parse the file path and extract model id, dataset path, dataset name and split
    def parse_filepath(fp: str):
        model_index = fp.find("MODEL_")
        fp = fp[model_index:]
        ds_index = fp.find("DATASET_")
        model_id = fp[:ds_index].replace("MODEL_", "").rstrip("_")
        author_index = model_id.find("-")
        model_id = model_id[:author_index] + "/" + model_id[author_index + 1 :]

        ds_fp = fp[ds_index:]
        dataset_id = ds_fp.replace("DATASET_", "").rstrip(".jsonl")
        return model_id, dataset_id

    # Compute results per dataset
    results = {}
    wer_metric = evaluate.load("wer")

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        references = [datum["text"] for datum in manifest]
        predictions = [datum["pred_text"] for datum in manifest]

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = wer

    print("*" * 80)
    print("Results per dataset:")
    print("*" * 80)

    for k, v in results.items():
        print(f"{k}: WER = {v:0.2f} %")

    # composite WER should be computed over all datasets and with the same key
    composite_wer = defaultdict(float)
    count_entries = defaultdict(int)
    for k, v in results.items():
        key = k.split("|")[0].strip()
        composite_wer[key] += v
        count_entries[key] += 1

    # normalize scores & print
    print()
    print("*" * 80)
    print("Composite WER:")
    print("*" * 80)
    for k, v in composite_wer.items():
        wer = v / count_entries[k]
        print(f"{k}: WER = {wer:0.2f} %")
    print("*" * 80)
    return composite_wer, results
