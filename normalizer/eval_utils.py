import os
import glob
import json
from difflib import SequenceMatcher

import evaluate
from collections import defaultdict


def normalize_compound_pairs(refs, preds):
    """Align compound word boundaries between ref/pred pairs.

    When a mismatch region has identical characters ignoring whitespace,
    normalize both sides to the joined form.
    """
    new_refs, new_preds = [], []
    for ref_text, pred_text in zip(refs, preds):
        ref_words = ref_text.split()
        pred_words = pred_text.split()

        sm = SequenceMatcher(None, ref_words, pred_words)
        new_rw, new_pw = [], []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                new_rw.extend(ref_words[i1:i2])
                new_pw.extend(pred_words[j1:j2])
            else:
                rc = "".join(ref_words[i1:i2])
                pc = "".join(pred_words[j1:j2])
                if rc == pc:
                    new_rw.append(rc)
                    new_pw.append(pc)
                else:
                    new_rw.extend(ref_words[i1:i2])
                    new_pw.extend(pred_words[j1:j2])

        new_refs.append(" ".join(new_rw))
        new_preds.append(" ".join(new_pw))
    return new_refs, new_preds


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
    audio_filepaths: list = None,
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
        audio_filepaths: List of file paths for each audio sample.
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
    if audio_filepaths is not None and len(audio_filepaths) != len(references):
        raise ValueError(
            f"The number of samples in `audio_filepaths` ({len(audio_filepaths)}) "
            f"must match `references` ({len(references)})."
        )

    # Filter out samples where the normalized reference is empty,
    # e.g. all-filler words removed by normalization. Mutates the caller's
    # lists in-place (via slice assignment) so downstream WER computation
    # in caller scripts also sees the filtered data.
    valid_indices = [
        i for i, ref in enumerate(references) if isinstance(ref, str) and ref.strip()
    ]
    n_filtered = len(references) - len(valid_indices)
    if n_filtered > 0:
        print(f"Filtered {n_filtered} empty references")
        references[:] = [references[i] for i in valid_indices]
        transcriptions[:] = [transcriptions[i] for i in valid_indices]
        if audio_length is not None:
            audio_length[:] = [audio_length[i] for i in valid_indices]
        if transcription_time is not None:
            transcription_time[:] = [transcription_time[i] for i in valid_indices]
        if audio_filepaths is not None:
            audio_filepaths[:] = [audio_filepaths[i] for i in valid_indices]

    audio_length = (
        audio_length if audio_length is not None else len(references) * [None]
    )
    transcription_time = (
        transcription_time
        if transcription_time is not None
        else len(references) * [None]
    )
    audio_filepaths = (
        audio_filepaths if audio_filepaths is not None else len(references) * [None]
    )

    basedir = "./results/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    manifest_path = os.path.join(
        basedir, f"MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}.jsonl"
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (text, transcript, audio_length, transcription_time, audio_filepath) in enumerate(
            zip(references, transcriptions, audio_length, transcription_time, audio_filepaths)
        ):
            datum = {
                "audio_filepath": audio_filepath if audio_filepath else f"sample_{idx}",
                "duration": audio_length,
                "time": transcription_time,
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path


def score_results(directory: str, model_id: str = None, multilingual: bool = False, csv_only: bool = False):
    """
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.
        multilingual: If True, apply compound word boundary normalization before
                      WER computation. Should only be enabled for non-English benchmarks.
        csv_only: If True, suppress all output except the CSV summary block.

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
    original_model_id = model_id  # preserve original (e.g. "distil-whisper/distil-large-v3.5") for CSV label
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
        dataset_id = ds_fp.replace("DATASET_", "").removesuffix(".jsonl")
        return model_id, dataset_id

    # Compute WER results per dataset, and RTFx over all datasets
    results = {}
    wer_metric = evaluate.load("wer")

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        manifest = [datum for datum in manifest if datum["text"].strip()]

        references = [datum["text"] for datum in manifest]
        predictions = [datum["pred_text"] for datum in manifest]

        time = [datum["time"] for datum in manifest]
        duration = [datum["duration"] for datum in manifest]
        compute_rtfx = all(time) and all(duration)

        if multilingual:
            references, predictions = normalize_compound_pairs(references, predictions)

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        if compute_rtfx:
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(sum(duration) / sum(time), 4)
        else:
            audio_length = inference_time = rtfx = None

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = {"wer": wer, "audio_length": audio_length, "inference_time": inference_time, "rtfx": rtfx}

    if not csv_only:
        print("*" * 80)
        print("Results per dataset:")
        print("*" * 80)

        for k, v in results.items():
            metrics = f"{k}: WER = {v['wer']:0.2f} %"
            if v["rtfx"] is not None:
                metrics += f", RTFx = {v['rtfx']:0.2f}"
            print(metrics)

    # composite WER should be computed over all datasets and with the same key
    composite_wer = defaultdict(float)
    composite_audio_length = defaultdict(float)
    composite_inference_time = defaultdict(float)
    count_entries = defaultdict(int)
    for k, v in results.items():
        key = k.split("|")[0].strip()
        composite_wer[key] += v["wer"]
        if v["rtfx"] is not None:
            composite_audio_length[key] += v["audio_length"]
            composite_inference_time[key] += v["inference_time"]
        else:
            composite_audio_length[key] = composite_inference_time[key] = None
        count_entries[key] += 1

    # normalize scores & print
    if not csv_only:
        print()
        print("*" * 80)
        print("Composite Results:")
        print("*" * 80)
        for k, v in composite_wer.items():
            wer = v / count_entries[k]
            print(f"{k}: WER = {wer:0.2f} %")
        for k in composite_audio_length:
            if composite_audio_length[k] is not None:
                rtfx = composite_audio_length[k] / composite_inference_time[k]
                print(f"{k}: RTFx = {rtfx:0.2f}")
        print("*" * 80)

    # ── Family definitions ────────────────────────────────────────────────────
    # Each entry: (family_key, presence_substring, header, col_map)
    # col_map: ds_substr → (column_label, group_or_None)
    FAMILY_CONFIGS = [
        (
            "appen",
            "appen",
            "model,Avg Appen WER,Avg Scripted,Avg Conversational,"
            "Scripted-US,Scripted-AU,Scripted-CA,Scripted-IN,"
            "Conversational-US003,Conversational-US004,Conversational-IN",
            {
                "appen_scripted_filtered__american":                     ("Scripted-US",          "scripted"),
                "appen_scripted_filtered__australian":                   ("Scripted-AU",          "scripted"),
                "appen_scripted_filtered__canadian":                     ("Scripted-CA",          "scripted"),
                "appen_scripted_filtered__indian":                       ("Scripted-IN",          "scripted"),
                "appen_conversational_segmented_filtered__american_003": ("Conversational-US003", "conversational"),
                "appen_conversational_segmented_filtered__american_004": ("Conversational-US004", "conversational"),
                "appen_conversational_segmented_filtered__indian":       ("Conversational-IN",    "conversational"),
            },
        ),
        (
            "dataocean",
            "dataocean",
            "model,Avg DataOcean WER,Avg Scripted,Avg Conversational,"
            "Scripted-US,Scripted-GB,Conversational-US,Conversational-GB",
            {
                "dataocean_scripted_filtered__en_US":                  ("Scripted-US",       "scripted"),
                "dataocean_scripted_filtered__en_GB":                  ("Scripted-GB",       "scripted"),
                "dataocean_conversational_segmented_filtered__en_US":  ("Conversational-US", "conversational"),
                "dataocean_conversational_segmented_filtered__en_GB":  ("Conversational-GB", "conversational"),
            },
        ),
        (
            "public",
            None,   # always printed when public datasets are present
            "model,RTFx,License,Size (B),# Languages,Encoder,Decoder,"
            "AMI WER,Earnings22 WER,Gigaspeech WER,LS Clean WER,LS Other WER,SPGISpeech WER,Voxpopuli WER",
            {
                "ami_test":               ("AMI WER",        None),
                "earnings22_test":        ("Earnings22 WER", None),
                "gigaspeech_test":        ("Gigaspeech WER", None),
                "librispeech_test.clean": ("LS Clean WER",   None),
                "librispeech_test.other": ("LS Other WER",   None),
                "spgispeech_test":        ("SPGISpeech WER", None),
                "voxpopuli_test":         ("Voxpopuli WER",  None),
            },
        ),
    ]

    all_dataset_ids = " ".join(results.keys())

    def find_wer_in(model_key, col_label, col_map):
        for ds_substr, (label, _group) in col_map.items():
            if label == col_label:
                for result_key, result_val in results.items():
                    if model_key.rstrip() in result_key and ds_substr in result_key:
                        return result_val["wer"]
        return None

    def print_csv_block(header, col_map, family_name=None):
        csv_columns = [lbl for lbl, _grp in col_map.values()]
        # deduplicate while preserving order
        seen = set()
        csv_columns = [c for c in csv_columns if not (c in seen or seen.add(c))]

        title = f"CSV Summary ({family_name}):" if family_name else "CSV Summary:"
        print()
        print("*" * 80)
        print(title)
        print("*" * 80)
        print(header)

        for model_key in composite_wer:
            csv_model_label = original_model_id if original_model_id is not None else model_key
            wer_vals = {col: find_wer_in(model_key, col, col_map) for col in csv_columns}
            wer_cols = [str(wer_vals[col]) if wer_vals[col] is not None else "" for col in csv_columns]

            is_private = any(grp is not None for _lbl, grp in col_map.values())
            if is_private:
                scripted_wers       = [v for _ds, (lbl, grp) in col_map.items()
                                        if grp == "scripted"       and (v := wer_vals.get(lbl)) is not None]
                conversational_wers = [v for _ds, (lbl, grp) in col_map.items()
                                        if grp == "conversational" and (v := wer_vals.get(lbl)) is not None]
                all_wers            = [v for v in wer_vals.values() if v is not None]
                avg_overall        = round(sum(all_wers) / len(all_wers), 2)            if all_wers else ""
                avg_scripted       = round(sum(scripted_wers) / len(scripted_wers), 2)  if scripted_wers else ""
                avg_conv           = round(sum(conversational_wers) / len(conversational_wers), 2) if conversational_wers else ""
                print(f"{csv_model_label},{avg_overall},{avg_scripted},{avg_conv}," + ",".join(wer_cols))
            else:
                if composite_audio_length[model_key] is not None:
                    rtfx_val = round(composite_audio_length[model_key] / composite_inference_time[model_key], 2)
                else:
                    rtfx_val = ""
                print(f"{csv_model_label},{rtfx_val},,,,,," + ",".join(wer_cols))

        print("*" * 80)

    # ── Print one CSV block per detected family ───────────────────────────────
    for family_key, presence_substr, header, col_map in FAMILY_CONFIGS:
        family_name = family_key.capitalize()  # "Appen", "Dataocean", "Public"
        # Public block: print only if at least one public dataset key is found
        if presence_substr is None:
            has_public = any(ds_substr in all_dataset_ids for ds_substr in col_map)
            if has_public:
                print_csv_block(header, col_map, family_name)
        else:
            if presence_substr in all_dataset_ids:
                print_csv_block(header, col_map, family_name)

    return composite_wer, results
