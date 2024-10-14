import os
import glob
import json
import os

import evaluate
import pandas as pd


def read_manifest(manifest_path: str):
    """
    Reads a manifest file (jsonl format) and returns a list of dictionaries containing samples.
    """
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 0:
                datum = json.loads(line)
                data.append(datum)
    return data


def write_manifest(
    references: list,
    transcriptions: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    audio_length: list = None,
    transcription_time: list = None,
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
        audio_length: Length of each audio sample in seconds.
        transcription_time: Transcription time of each sample in seconds.

    Returns:
        Path to the manifest file.
    """
    model_id = model_id.replace("/", "-")
    dataset_path = dataset_path.replace("/", "-")
    dataset_name = dataset_name.replace("/", "-")

    if len(references) != len(transcriptions):
        raise ValueError(
            f"The number of samples in `references` ({len(references)}) "
            f"must match `transcriptions` ({len(transcriptions)})."
        )

    if audio_length is not None and len(audio_length) != len(references):
        raise ValueError(
            f"The number of samples in `audio_length` ({len(audio_length)}) "
            f"must match `references` ({len(references)})."
        )
    if transcription_time is not None and len(transcription_time) != len(references):
        raise ValueError(
            f"The number of samples in `transcription_time` ({len(transcription_time)}) "
            f"must match `references` ({len(references)})."
        )

    audio_length = (
        audio_length if audio_length is not None else len(references) * [None]
    )
    transcription_time = (
        transcription_time
        if transcription_time is not None
        else len(references) * [None]
    )

    basedir = "./results/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    manifest_path = os.path.join(
        basedir, f"MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}.jsonl"
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (text, transcript, audio_length, transcription_time) in enumerate(
            zip(references, transcriptions, audio_length, transcription_time)
        ):
            datum = {
                "audio_filepath": f"sample_{idx}",  # dummy value for Speech Data Processor
                "duration": audio_length,
                "time": transcription_time,
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path


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

    # Compute WER results per dataset, and RTFx over all datasets
    wer_metric = evaluate.load("wer")

    results_df = pd.DataFrame()
    for result_file in result_files:
        manifest = read_manifest(result_file)

        model_id, dataset_id = parse_filepath(result_file)

        # if dataset_id not in results_df.columns:
        wer_column = dataset_id + "_wer"
        rtfx_column = dataset_id + "_RTFx"

        results_df[wer_column] = (
            None if wer_column not in results_df.columns else results_df[wer_column]
        )
        results_df[rtfx_column] = (
            None if rtfx_column not in results_df.columns else results_df[rtfx_column]
        )

        references = [datum["text"] for datum in manifest]
        predictions = [datum["pred_text"] for datum in manifest]

        time = [datum["time"] for datum in manifest]
        duration = [datum["duration"] for datum in manifest]
        compute_rtfx = all(time) and all(duration)

        rtfx = round(sum(duration) / sum(time), 4) if compute_rtfx else None

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        results_df.loc[model_id, wer_column] = wer
        results_df.loc[model_id, rtfx_column] = rtfx

    results_df["average_wer"] = round(results_df.filter(like="_wer").mean(axis=1), 2)
    results_df["average_rtfx"] = round(results_df.filter(like="_RTFx").mean(axis=1), 2)

    results_df.to_csv("eval_results.csv")

    eval_filename = "eval_results.csv"
    
    print(f"Results saved to {eval_filename}")
