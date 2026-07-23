import os
import glob
import json
from difflib import SequenceMatcher

from collections import defaultdict
from kaldialign import batch_error_rate


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


def score_results(directory: str, model_id: str = None, multilingual: bool = False, csv_only: bool = False, language: str = "en", families: list = None):
    """
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.
        multilingual: If True, apply compound word boundary normalization before
                      WER computation. Should only be enabled for non-English benchmarks.
        csv_only: If True, suppress all output except the CSV summary block.
        language: Language code used for normalization (e.g. 'en', 'de', 'fr').
                  When not 'en', ml_normalizer is used instead of the English normalizer.
        families: Optional list of family keys ("appen", "dataocean", "public", "extra",
                  "ml_de", "ml_fr", "ml_it", "ml_es", "ml_pt") restricting which CSV
                  summary blocks are printed. None prints all detected families.

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
        result_files = [
            fp for fp in result_files
            if f"/{model_id}/" in fp or f"MODEL_{model_id}_DATASET_" in fp
        ]

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
            "AMI-Cleaned WER,Earnings22 WER,Gigaspeech-Cleaned WER,LS Clean WER,LS Other WER,SPGISpeech WER,Voxpopuli-Cleaned-AA WER",
            {
                
                "ami_cleaned_test":          ("AMI-Cleaned WER",        None),
                "earnings22_test":           ("Earnings22 WER", None),
                "gigaspeech_cleaned_test":   ("Gigaspeech-Cleaned WER", None),
                "librispeech_test.clean":    ("LS Clean WER",   None),
                "librispeech_test.other":    ("LS Other WER",   None),
                "spgispeech_test":           ("SPGISpeech WER", None),
                "voxpopuli_cleaned_aa_test": ("Voxpopuli-Cleaned-AA WER",  None),
            },
        ),
        (
            "extra",
            "_cleaned",
            "model,AMI WER,Gigaspeech WER,Voxpopuli WER",
            {
                "ami_test":               ("AMI WER",        None),
                "gigaspeech_test":        ("Gigaspeech WER", None),
                "voxpopuli_test":         ("Voxpopuli WER",  None),
            },
        ),
    ]

    # Multilingual families: one per language, covering whichever of
    # FLEURS / MCV / MLS include that language.
    ML_LANG_DATASETS = {
        "de": ["fleurs", "mcv"],
        "fr": ["fleurs", "mcv", "mls"],
        "it": ["fleurs", "mcv", "mls"],
        "es": ["fleurs", "mcv", "mls"],
        "pt": ["fleurs", "mls"],
    }
    ML_DATASET_LABELS = {"fleurs": "FLEURS", "mcv": "MCV", "mls": "MLS"}
    for lang, datasets in ML_LANG_DATASETS.items():
        col_map = {
            f"{dataset}_{lang}_test": (f"{ML_DATASET_LABELS[dataset]} WER", None)
            for dataset in datasets
        }
        header = "model,RTFx," + ",".join(
            f"{ML_DATASET_LABELS[dataset]} WER" for dataset in datasets
        )
        FAMILY_CONFIGS.append((f"ml_{lang}", f"_{lang}_test", header, col_map))

    # Restrict scoring to only the datasets relevant to the requested families.
    # Without this, files outside the requested families would still be scored
    # (and printed in the "Results per dataset"/"Composite Results" sections)
    # using whichever `language` normalizer was passed for this call, which is
    # wrong for unrelated-language datasets that happen to share the directory.
    if families is not None:
        allowed_substrs = []
        for family_key, presence_substr, _header, col_map in FAMILY_CONFIGS:
            if family_key in families:
                if presence_substr is not None:
                    allowed_substrs.append(presence_substr)
                else:
                    allowed_substrs.extend(col_map.keys())
        result_files = [
            fp for fp in result_files
            if any(substr in parse_filepath(fp)[1] for substr in allowed_substrs)
        ]
        if len(result_files) == 0:
            raise ValueError(f"No result files found in {directory} matching families {families}")

    # Compute WER results per dataset, and RTFx over all datasets
    from normalizer import data_utils  # deferred to avoid circular import
    results = {}

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        if language == "en":
            normalize = data_utils.normalizer
        else:
            normalize = lambda t: data_utils.ml_normalizer(t, lang=language)
        references = [normalize(datum["text"]) for datum in manifest]
        predictions = [normalize(datum["pred_text"]) for datum in manifest]

        time = [datum["time"] for datum in manifest]
        duration = [datum["duration"] for datum in manifest]
        compute_rtfx = all(time) and all(duration)

        if multilingual:
            # Align compound word boundaries (e.g. German/Italian compounds)
            # before scoring, so split-vs-joined spelling doesn't count as an error.
            references, predictions = normalize_compound_pairs(references, predictions)

        # Use kaldialign batch_error_rate with merge_compounds=True so that
        # split compounds (e.g. "white paper" vs "whitepaper") count as
        # 0 errors in either direction.
        refs_split  = [tuple(r.split()) for r in references]
        preds_split = [tuple(p.split()) for p in predictions]
        r = batch_error_rate(refs_split, preds_split, merge_compounds=True)
        total_ins, total_del, total_sub = r["ins"], r["del"], r["sub"]
        wer = r["err_rate"]
        extra = {"ins": total_ins, "del": total_del, "sub": total_sub}

        wer = round(100 * wer, 2)

        if compute_rtfx:
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(sum(duration) / sum(time), 4)
        else:
            audio_length = inference_time = rtfx = None

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = {"wer": wer, "audio_length": audio_length, "inference_time": inference_time, "rtfx": rtfx, **extra}

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

    all_dataset_ids = " ".join(results.keys())

    def find_wer_in(model_key, col_label, col_map):
        for ds_substr, (label, _group) in col_map.items():
            if label == col_label:
                for result_key, result_val in results.items():
                    if model_key.rstrip() in result_key and ds_substr in result_key:
                        return result_val["wer"]
        return None

    def print_csv_block(header, col_map, family_key=None, family_name=None):
        csv_columns = [lbl for lbl, _grp in col_map.values()]
        # deduplicate while preserving order
        seen = set()
        csv_columns = [c for c in csv_columns if not (c in seen or seen.add(c))]

        title = f"CSV Summary ({family_name}):" if family_name else "CSV Summary:"
        print()
        print("*" * 80)
        print(title)
        print("*" * 80)

        if len(composite_wer) == 1:
            for model_key in composite_wer:
                wer_vals = [find_wer_in(model_key, col, col_map) for col in csv_columns]
                wer_vals = [v for v in wer_vals if v is not None]
                if wer_vals:
                    avg = round(sum(wer_vals) / len(wer_vals), 2)
                    label = original_model_id if original_model_id is not None else model_key.strip()
                    print(f"avg WER ({label}) = {avg}")

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
                n_prefix = len(header.split(',')) - 1 - len(csv_columns)
                if family_key == "public" or (family_key or "").startswith("ml_"):
                    family_audio = sum(
                        results[rk]["audio_length"]
                        for ds_substr in col_map
                        for rk in results
                        if model_key.rstrip() in rk and ds_substr in rk and results[rk]["audio_length"] is not None
                    )
                    family_time = sum(
                        results[rk]["inference_time"]
                        for ds_substr in col_map
                        for rk in results
                        if model_key.rstrip() in rk and ds_substr in rk and results[rk]["inference_time"] is not None
                    )
                    rtfx_val = round(family_audio / family_time, 2) if family_time else ""
                    prefix_cols = [str(rtfx_val)] + [""] * (n_prefix - 1)
                else:
                    prefix_cols = [""] * n_prefix
                print(",".join([csv_model_label] + prefix_cols + wer_cols))

        print("*" * 80)

    # ── Print one CSV block per detected family ───────────────────────────────
    for family_key, presence_substr, header, col_map in FAMILY_CONFIGS:
        if families is not None and family_key not in families:
            continue
        if family_key.startswith("ml_"):
            family_name = family_key[len("ml_"):]  # "de", "fr", "it", "es", "pt"
        else:
            family_name = family_key.capitalize()  # "Appen", "Dataocean", "Public", "Extra"
        # Public block: print only if at least one public dataset key is found
        if presence_substr is None:
            has_public = any(ds_substr in all_dataset_ids for ds_substr in col_map)
            if has_public:
                print_csv_block(header, col_map, family_key, family_name)
        else:
            if presence_substr in all_dataset_ids:
                print_csv_block(header, col_map, family_key, family_name)

    return composite_wer, results
