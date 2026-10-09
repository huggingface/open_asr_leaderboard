import glob
import hashlib
import json
import os
from collections import defaultdict
from difflib import SequenceMatcher

from concurrent.futures import ProcessPoolExecutor

from kaldialign import edit_distance

# Languages scored with voi_oiwer (Orthographically Informed WER over a
# reference lattice) instead of plain WER. Maps language code → voi_oiwer
# input_language name.
OIWER_LANGUAGES = {
    "hi": "hindi",
}


def score_oiwer(manifest: list, language_name: str):
    """Score a manifest with voi_oiwer (lattice-based, orthography-aware WER).

    Each manifest entry must have "pred_text" and a reference in one of:
      - "reference_lists": the lattice — list of slots, each a list of
        accepted variants (a variant may span multiple words);
      - "text" as a JSON-encoded lattice (list of lists of str);
      - "text" as a plain string, converted to a trivial single-variant
        lattice (one slot per word).

    voi_oiwer applies its own indicnlp-based normalization internally, so no
    external normalizer should be applied beforehand.

    Returns (err_rate, total_ins, total_del, total_sub) with err_rate in [0, 1].
    """
    from voi_oiwer import oiwer  # deferred import: only needed for OIWER languages

    total_ins = total_del = total_sub = 0
    total_ref_words = 0
    for datum in manifest:
        reference_lists = datum.get("reference_lists")
        if reference_lists is None:
            text = datum["text"]
            if isinstance(text, str) and text.lstrip().startswith("[["):
                try:
                    text = json.loads(text)
                except json.JSONDecodeError:
                    pass
            if isinstance(text, list):
                reference_lists = text
            else:
                # Plain string reference: trivial lattice, one slot per word.
                reference_lists = [[word] for word in str(text).split()]

        _score, _h, _r, _ops, (ins, dele, sub), ref_words, _meta, _std = oiwer(
            hypothesis=datum["pred_text"],
            reference_lists=reference_lists,
            input_language=language_name,
        )
        total_ins += ins
        total_del += dele
        total_sub += sub
        total_ref_words += ref_words

    err_rate = (total_ins + total_del + total_sub) / total_ref_words if total_ref_words else 0.0
    return err_rate, total_ins, total_del, total_sub


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
    extra_fields: dict = None,
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
        extra_fields: Optional mapping of column name to a per-sample list, written
            alongside the standard fields.
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
    extra_fields = extra_fields or {}
    for field_name, values in extra_fields.items():
        if len(values) != len(references):
            raise ValueError(
                f"The number of samples in `{field_name}` ({len(values)}) "
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
        for idx, (
            text,
            transcript,
            audio_length,
            transcription_time,
            audio_filepath,
        ) in enumerate(
            zip(
                references,
                transcriptions,
                audio_length,
                transcription_time,
                audio_filepaths,
            )
        ):
            datum = {
                "audio_filepath": audio_filepath if audio_filepath else f"sample_{idx}",
                "duration": audio_length,
                "time": transcription_time,
                "text": text,
                "pred_text": transcript,
                **{name: values[idx] for name, values in extra_fields.items()},
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path


# Per-manifest score files: `<manifest stem>.score.json`, next to the manifest.
SCORE_SUFFIX = ".score.json"


def score_file_path(manifest_path: str) -> str:
    return manifest_path.removesuffix(".jsonl") + SCORE_SUFFIX


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_score_file(manifest_path: str, language: str = "en", multilingual: bool = False):
    """Return the stored score of `manifest_path`, or None if absent or stale."""
    path = score_file_path(manifest_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            score = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        score.get("language") != language
        or score.get("multilingual") != multilingual
        or score.get("manifest_sha256") != _sha256(manifest_path)
    ):
        return None
    return score


def write_score_file(manifest_path: str, score: dict, language: str = "en", multilingual: bool = False) -> str:
    """Store `score` (as returned by score_manifest) next to `manifest_path`."""
    path = score_file_path(manifest_path)
    record = {
        **score,
        "language": language,
        "multilingual": multilingual,
        "manifest_sha256": _sha256(manifest_path),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return path


def available_cpus() -> int:
    """CPUs this process may run on (respects taskset / cgroup CPU sets on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def _normalize(text: str, language: str) -> str:
    from normalizer import data_utils  # deferred to avoid circular import

    if language == "en":
        return data_utils.normalizer(text)
    return data_utils.ml_normalizer(text, lang=language)


def _align_pair(job):
    """Normalize one (reference, prediction) pair and align it, as score_results scores it."""
    ref, pred, language, multilingual = job
    ref, pred = _normalize(ref, language), _normalize(pred, language)
    if multilingual:
        # Align compound word boundaries (e.g. German/Italian compounds)
        # before scoring, so split-vs-joined spelling doesn't count as an error.
        (ref,), (pred,) = normalize_compound_pairs([ref], [pred])
    # kaldialign with merge_compounds=True, so that split compounds (e.g.
    # "white paper" vs "whitepaper") count as 0 errors in either direction.
    return edit_distance(tuple(ref.split()), tuple(pred.split()), merge_compounds=True)


def align_manifests(manifests: list, language: str = "en", multilingual: bool = False,
                    num_workers: int = None) -> list:
    """Error counts of each manifest: {"ins", "del", "sub", "total", "ref_len", "err_rate"}.

    The same counts as batch_error_rate(..., merge_compounds=True) over the normalized
    texts, which just sums edit_distance over the pairs. Here the pairs of *all* the
    manifests -- normalization included -- go through one process pool: alignment is
    quadratic in the length of a recording, hour-long transcripts take tens of seconds
    each, and pooling across manifests means the run waits for its single longest
    recording rather than for the longest of each manifest in turn.

    num_workers: processes to use; None uses every CPU available to this process, 1
    works serially in this process. Never more than there are pairs.
    """
    jobs = [
        (m, (datum["text"], datum["pred_text"], language, multilingual))
        for m, manifest in enumerate(manifests)
        for datum in manifest
    ]
    workers = min(num_workers or available_cpus(), len(jobs))
    # Longest first, so that a long recording does not start last and hold up the pool.
    jobs.sort(key=lambda job: len(job[1][0]) * len(job[1][1]), reverse=True)
    if workers <= 1:
        results = map(_align_pair, [job for _, job in jobs])
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        # chunksize=1: the pairs differ in cost by orders of magnitude, so they are
        # handed out one at a time for the pool to balance itself.
        results = pool.map(_align_pair, [job for _, job in jobs], chunksize=1)
    totals = [{"ins": 0, "del": 0, "sub": 0, "total": 0, "ref_len": 0} for _ in manifests]
    try:
        for (m, _), cur in zip(jobs, results):
            for key in totals[m]:
                totals[m][key] += cur[key]
    finally:
        if pool is not None:
            pool.shutdown()
    for t in totals:
        if t["ref_len"]:
            t["err_rate"] = t["total"] / t["ref_len"]
        else:
            t["err_rate"] = 0.0 if t["total"] == 0 else float("inf")
    return totals


def score_manifests(
    manifests: list, language: str = "en", multilingual: bool = False, num_workers: int = None
) -> list:
    """WER (in %) and RTFx of each manifest, as score_results computes them.

    Each score is {"wer", "ins", "del", "sub", "audio_length", "inference_time", "rtfx"};
    the last three are None when the manifest has no timing information. The
    manifests share one process pool (see align_manifests).
    """
    manifests = [merge_chunked_manifest(m) for m in manifests]
    if language in OIWER_LANGUAGES:
        # Lattice-based, orthography-aware scoring (voi_oiwer). The package
        # applies its own indicnlp-based normalization internally.
        counts = [score_oiwer(m, OIWER_LANGUAGES[language]) for m in manifests]
    else:
        counts = [
            (r["err_rate"], r["ins"], r["del"], r["sub"])
            for r in align_manifests(manifests, language, multilingual, num_workers)
        ]

    scores = []
    for manifest, (wer, total_ins, total_del, total_sub) in zip(manifests, counts):
        time = [datum["time"] for datum in manifest]
        duration = [datum["duration"] for datum in manifest]
        if all(time) and all(duration):
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(audio_length / inference_time, 4)
        else:
            audio_length = inference_time = rtfx = None
        scores.append({
            "wer": round(100 * wer, 2),
            "ins": total_ins,
            "del": total_del,
            "sub": total_sub,
            "audio_length": audio_length,
            "inference_time": inference_time,
            "rtfx": rtfx,
        })
    return scores


def score_manifest(
    manifest: list, language: str = "en", multilingual: bool = False, num_workers: int = None
) -> dict:
    """score_manifests for a single manifest."""
    return score_manifests([manifest], language, multilingual, num_workers)[0]


CHUNK_PARENT_KEY = "parent_id"
CHUNK_INDEX_KEY = "chunk_index"


def merge_chunked_manifest(manifest: list):
    """Collapse per-chunk result rows into one row per parent session.

    Chunks are non-overlapping and cover the session in order, so predictions are
    concatenated by `chunk_index` and scored against the session reference.
    Durations and times are summed, leaving RTFx unaffected.

    Manifests without a `parent_id` field are returned unchanged.
    """
    if not manifest or CHUNK_PARENT_KEY not in manifest[0]:
        return manifest

    sessions = defaultdict(list)
    for datum in manifest:
        sessions[datum[CHUNK_PARENT_KEY]].append(datum)

    def _sum_or_none(values):
        return sum(values) if all(v is not None for v in values) else None

    merged = []
    for parent_id in sorted(sessions):
        chunks = sorted(sessions[parent_id], key=lambda d: d[CHUNK_INDEX_KEY])

        indices = [chunk[CHUNK_INDEX_KEY] for chunk in chunks]
        if indices != list(range(len(indices))):
            print(
                f"WARNING: chunk indices for session {parent_id} are not "
                f"contiguous from 0 ({indices}); some chunks may be missing, "
                "which will inflate the deletion count."
            )

        merged.append(
            {
                "audio_filepath": parent_id,
                "duration": _sum_or_none([chunk["duration"] for chunk in chunks]),
                "time": _sum_or_none([chunk["time"] for chunk in chunks]),
                "text": chunks[0]["text"],
                "pred_text": " ".join(
                    pred
                    for chunk in chunks
                    if (pred := (chunk["pred_text"] or "").strip())
                ),
            }
        )
    return merged


def score_results(
    directory: str,
    model_id: str = None,
    multilingual: bool = False,
    csv_only: bool = False,
    language: str = "en",
    families: list = None,
    use_cache: bool = False,
    recompute: bool = False,
    written_scores: list = None,
    num_workers: int = None,
):
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
                  Languages in OIWER_LANGUAGES (e.g. 'hi') are scored with
                  voi_oiwer over a reference lattice instead of plain WER.
        families: Optional list of family keys ("appen", "dataocean", "voicearena_private",
                  "voicearena_private_hi", "public", "extra", "longform", "ml_de", "ml_fr", "ml_it", "ml_es",
                  "ml_pt", "ml_nl", "ml_hy") restricting which CSV summary blocks are printed.
                  None prints all detected families.
        use_cache: If True, read each manifest's score from its `.score.json` file
                   (see SCORE_SUFFIX) when one matches, and write one when not.
        recompute: With use_cache, ignore existing score files and rewrite them,
                   e.g. after a change to the normalizer.
        written_scores: Optional list; the path of every score file written is
                   appended to it, so the caller can upload them.
        num_workers: Processes used to align each manifest; None uses every CPU
                   available, 1 aligns serially. The manifests without a stored
                   score share one pool; see align_manifests.

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
            fp
            for fp in result_files
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

    # CORAAL subsets (bezzam/coraal), macro-averaged into a single long-form column.
    CORAAL_SPLITS = ["ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD"]

    # ── Family definitions ────────────────────────────────────────────────────
    # Each entry: (family_key, presence_substring, header, col_map)
    # col_map: ds_substr → (column_label, group_or_None)
    FAMILY_CONFIGS = [
        (
            "appen",
            "appen",
            "model,Avg WER,Avg Scripted,Avg Conversational,"
            "Scripted-US,Scripted-AU,Scripted-CA,Scripted-IN,"
            "Conversational-US003,Conversational-US004,Conversational-IN",
            {
                "appen_scripted_filtered__american": ("Scripted-US", "scripted"),
                "appen_scripted_filtered__australian": ("Scripted-AU", "scripted"),
                "appen_scripted_filtered__canadian": ("Scripted-CA", "scripted"),
                "appen_scripted_filtered__indian": ("Scripted-IN", "scripted"),
                "appen_conversational_segmented_filtered__american_003": (
                    "Conversational-US003",
                    "conversational",
                ),
                "appen_conversational_segmented_filtered__american_004": (
                    "Conversational-US004",
                    "conversational",
                ),
                "appen_conversational_segmented_filtered__indian": (
                    "Conversational-IN",
                    "conversational",
                ),
            },
        ),
        (
            "dataocean",
            "dataocean",
            "model,Avg DataOcean WER,Avg Scripted,Avg Conversational,"
            "Scripted-US,Scripted-GB,Conversational-US,Conversational-GB",
            {
                "dataocean_scripted_filtered__en_US": ("Scripted-US", "scripted"),
                "dataocean_scripted_filtered__en_GB": ("Scripted-GB", "scripted"),
                "dataocean_conversational_segmented_filtered__en_US": (
                    "Conversational-US",
                    "conversational",
                ),
                "dataocean_conversational_segmented_filtered__en_GB": (
                    "Conversational-GB",
                    "conversational",
                ),
            },
        ),
        (
            "voicearena_private",
            "HF_English",
            "model,HF_English_Private_Set",
            {
                "HF_English_Private_Set__test": ("HF_English_Private_Set", None),
            },
        ),
        (
            "voicearena_private_hi",
            "HF_Hindi_Private_Set",
            "model,HF_Hindi_Private_Set",
            {
                "HF_Hindi_Private_Set__test": ("HF_Hindi_Private_Set", None),
            },
        ),
        (
            "public",
            None,  # always printed when public datasets are present
            "model,avg,RTFx,License,Size (B),# Languages,Encoder,Decoder,Training data disclosure,"
            "AMI-Cleaned WER,Earnings22-Cleaned-AA-chunked WER,Gigaspeech-Cleaned WER,LS Clean WER,LS Other WER,SPGISpeech WER,URGENT2024 WER,Voice Arena Monsoon WER,Voxpopuli-AA-Cleaned WER",
            # Keys of hf-audio/open-asr-leaderboard configs carry the repo slug, so
            # that a same-named set from another repo (e.g. the long-form
            # asr-leaderboard-longform_earnings22_test) is not picked up as well.
            {
                "open-asr-leaderboard_ami_cleaned_test": ("AMI-Cleaned WER", None),
                # Datasets in their own repo are run without a config name, so their
                # manifest id is "<repo-slug>__<split>". The name-based key is kept
                # for manifests produced before that convention.
                "Earnings22-Cleaned-AA-chunked__test": (
                    "Earnings22-Cleaned-AA-chunked WER",
                    None,
                ),
                "earnings22_cleaned_aa_chunked_test": (
                    "Earnings22-Cleaned-AA-chunked WER",
                    None,
                ),
                "open-asr-leaderboard_gigaspeech_cleaned_test": ("Gigaspeech-Cleaned WER", None),
                "open-asr-leaderboard_librispeech_test.clean": ("LS Clean WER", None),
                "open-asr-leaderboard_librispeech_test.other": ("LS Other WER", None),
                "open-asr-leaderboard_spgispeech_test": ("SPGISpeech WER", None),
                "open-asr-leaderboard_urgent2024_test": ("URGENT2024 WER", None),
                "Monsoon_en_IN_test__test": ("Voice Arena Monsoon WER", None),
                "open-asr-leaderboard_voxpopuli_cleaned_aa_test": ("Voxpopuli-AA-Cleaned WER", None),
            },
        ),
        (
            "extra",
            None,  # printed when any of its datasets is present
            "model,AMI WER,Earnings22 WER,Gigaspeech WER,Voxpopuli WER,"
            "URGENT2024-Clean WER",
            {
                "open-asr-leaderboard_ami_test": ("AMI WER", None),
                "open-asr-leaderboard_earnings22_test": ("Earnings22 WER", None),
                "open-asr-leaderboard_gigaspeech_test": ("Gigaspeech WER", None),
                "open-asr-leaderboard_voxpopuli_test": ("Voxpopuli WER", None),
                "open-asr-leaderboard_urgent2024_clean_test": ("URGENT2024-Clean WER", None),
            },
        ),
        (
            "longform",
            None,
            "model,avg,RTFx,earnings21,earnings22,coraal_avg,"
            + ",".join(f"coraal_{split}" for split in CORAAL_SPLITS),
            {
                "asr-leaderboard-longform_earnings21_test": ("earnings21", None),
                "asr-leaderboard-longform_earnings22_test": ("earnings22", None),
                **{
                    f"coraal_{split}_test": (f"coraal_{split}", None)
                    for split in CORAAL_SPLITS
                },
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
        "nl": ["fleurs", "mcv", "mls"],
        "hy": ["fleurs", "mcv"],
        # Hindi: VoiceArena/Monsoon_hi_test (scored with voi_oiwer, see OIWER_LANGUAGES)
        "hi": ["Monsoon"],
    }
    ML_DATASET_LABELS = {
        "fleurs": "FLEURS",
        "mcv": "MCV",
        "mls": "MLS",
        "Monsoon": "Monsoon",
    }
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
            fp
            for fp in result_files
            if any(substr in parse_filepath(fp)[1] for substr in allowed_substrs)
        ]
        if len(result_files) == 0:
            raise ValueError(
                f"No result files found in {directory} matching families {families}"
            )

    # Compute WER results per dataset, and RTFx over all datasets
    results = {}

    scores = {}
    if use_cache and not recompute:
        for result_file in result_files:
            score = read_score_file(result_file, language, multilingual)
            if score is not None:
                scores[result_file] = score
    # Everything without a stored score is scored in one go, so that the alignments
    # of all the manifests share one process pool.
    to_score = [fp for fp in result_files if fp not in scores]
    if to_score:
        fresh = score_manifests(
            [read_manifest(fp) for fp in to_score], language, multilingual, num_workers
        )
        for result_file, score in zip(to_score, fresh):
            scores[result_file] = score
            if use_cache:
                path = write_score_file(result_file, score, language, multilingual)
                if written_scores is not None:
                    written_scores.append(path)

    for result_file in result_files:
        model_id_of_file, dataset_id = parse_filepath(result_file)
        score = scores[result_file]

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = {
            key: score[key]
            for key in ("wer", "audio_length", "inference_time", "rtfx", "ins", "del", "sub")
        }

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

    def find_metric_in(model_key, col_label, col_map, metric="wer"):
        for ds_substr, (label, _group) in col_map.items():
            if label == col_label:
                for result_key, result_val in results.items():
                    if model_key.rstrip() in result_key and ds_substr in result_key:
                        return result_val[metric]
        return None

    def find_wer_in(model_key, col_label, col_map):
        return find_metric_in(model_key, col_label, col_map, "wer")

    def longform_averages(wer_vals):
        """Return (coraal_avg, avg) for the longform family, over the columns present."""
        coraal = [
            v for lbl, v in wer_vals.items() if lbl.startswith("coraal_") and v is not None
        ]
        coraal_avg = round(sum(coraal) / len(coraal), 2) if coraal else None
        parts = [
            v
            for v in (wer_vals.get("earnings21"), wer_vals.get("earnings22"), coraal_avg)
            if v is not None
        ]
        avg = round(sum(parts) / len(parts), 2) if parts else None
        return coraal_avg, avg

    def print_csv_block(
        header, col_map, family_key=None, family_name=None, per_dataset_rtfx=False
    ):
        csv_columns = [lbl for lbl, _grp in col_map.values()]
        # deduplicate while preserving order
        seen = set()
        csv_columns = [c for c in csv_columns if not (c in seen or seen.add(c))]

        # Prefix columns (RTFx, License, ...) are whatever the header has before
        # the per-dataset WER labels; measure it before appending anything.
        n_prefix = len(header.split(",")) - 1 - len(csv_columns)
        rtfx_columns = []
        if per_dataset_rtfx:
            # Derived from the WER labels rather than spelled out in the header,
            # so the two halves cannot drift out of order.
            rtfx_columns = [c.replace(" WER", " RTFx") for c in csv_columns]
            header = header + "," + ",".join(rtfx_columns)

        title = f"CSV Summary ({family_name}):" if family_name else "CSV Summary:"
        print()
        print("*" * 80)
        print(title)
        print("*" * 80)

        if len(composite_wer) == 1:
            for model_key in composite_wer:
                wer_vals = {col: find_wer_in(model_key, col, col_map) for col in csv_columns}
                if family_key == "longform":
                    avg = longform_averages(wer_vals)[1]
                else:
                    present = [v for v in wer_vals.values() if v is not None]
                    avg = round(sum(present) / len(present), 2) if present else None
                if avg is not None:
                    label = (
                        original_model_id
                        if original_model_id is not None
                        else model_key.strip()
                    )
                    print(f"avg WER ({label}) = {avg}")

        print(header)

        for model_key in composite_wer:
            csv_model_label = (
                original_model_id if original_model_id is not None else model_key
            )
            # Labels such as "org/model (fast-gpu-asr, ctc_greedy_search)" carry a
            # comma; quote them so the row keeps the header's field count.
            if any(ch in csv_model_label for ch in ',"\n'):
                csv_model_label = '"' + csv_model_label.replace('"', '""') + '"'
            wer_vals = {
                col: find_wer_in(model_key, col, col_map) for col in csv_columns
            }
            wer_cols = [
                str(wer_vals[col]) if wer_vals[col] is not None else ""
                for col in csv_columns
            ]
            if rtfx_columns:
                rtfx_vals = [
                    find_metric_in(model_key, col, col_map, "rtfx")
                    for col in csv_columns
                ]
                wer_cols += [str(v) if v is not None else "" for v in rtfx_vals]

            is_private = any(grp is not None for _lbl, grp in col_map.values())
            if is_private:
                scripted_wers = [
                    v
                    for _ds, (lbl, grp) in col_map.items()
                    if grp == "scripted" and (v := wer_vals.get(lbl)) is not None
                ]
                conversational_wers = [
                    v
                    for _ds, (lbl, grp) in col_map.items()
                    if grp == "conversational" and (v := wer_vals.get(lbl)) is not None
                ]
                all_wers = [v for v in wer_vals.values() if v is not None]
                avg_overall = (
                    round(sum(all_wers) / len(all_wers), 2) if all_wers else ""
                )
                avg_scripted = (
                    round(sum(scripted_wers) / len(scripted_wers), 2)
                    if scripted_wers
                    else ""
                )
                avg_conv = (
                    round(sum(conversational_wers) / len(conversational_wers), 2)
                    if conversational_wers
                    else ""
                )
                print(
                    f"{csv_model_label},{avg_overall},{avg_scripted},{avg_conv},"
                    + ",".join(wer_cols)
                )
            else:
                if family_key in ("public", "longform") or (family_key or "").startswith("ml_"):
                    family_audio = sum(
                        results[rk]["audio_length"]
                        for ds_substr in col_map
                        for rk in results
                        if model_key.rstrip() in rk
                        and ds_substr in rk
                        and results[rk]["audio_length"] is not None
                    )
                    family_time = sum(
                        results[rk]["inference_time"]
                        for ds_substr in col_map
                        for rk in results
                        if model_key.rstrip() in rk
                        and ds_substr in rk
                        and results[rk]["inference_time"] is not None
                    )
                    rtfx_val = (
                        round(family_audio / family_time, 2) if family_time else ""
                    )
                if family_key == "longform":
                    coraal_avg, avg = longform_averages(wer_vals)
                    cols = [avg, rtfx_val, wer_vals["earnings21"], wer_vals["earnings22"], coraal_avg]
                    cols += [wer_vals[f"coraal_{split}"] for split in CORAAL_SPLITS]
                    print(
                        ",".join(
                            [csv_model_label]
                            + ["" if v is None else str(v) for v in cols]
                        )
                    )
                    continue
                if family_key == "public" or (family_key or "").startswith("ml_"):
                    # Fill the prefix columns by name, not by position: the
                    # families do not share a prefix layout (ml_* is just
                    # "model,RTFx,...", public also carries avg and the metadata
                    # columns), and the metadata ones are filled in by hand later.
                    all_wers = [v for v in wer_vals.values() if v is not None]
                    known = {
                        "RTFx": rtfx_val,
                        # Unrounded, to match english_short_latest.csv, whose avg
                        # is the plain mean of the per-dataset WERs.
                        "avg": (sum(all_wers) / len(all_wers)) if all_wers else "",
                    }
                    prefix_labels = header.split(",")[1 : 1 + n_prefix]
                    prefix_cols = [str(known.get(lbl, "")) for lbl in prefix_labels]
                else:
                    prefix_cols = [""] * n_prefix
                print(",".join([csv_model_label] + prefix_cols + wer_cols))

        print("*" * 80)

    # ── Print one CSV block per detected family ───────────────────────────────
    for family_key, presence_substr, header, col_map in FAMILY_CONFIGS:
        if families is not None and family_key not in families:
            continue
        if family_key.startswith("ml_"):
            family_name = family_key[len("ml_") :]  # "de", "fr", "it", "es", "pt", "nl", "hy"
        else:
            family_name = (
                family_key.capitalize()
            )  # "Appen", "Dataocean", "Public", "Extra"
        # Public block: print only if at least one public dataset key is found
        if presence_substr is None:
            has_public = any(ds_substr in all_dataset_ids for ds_substr in col_map)
            if has_public:
                print_csv_block(
                    header,
                    col_map,
                    family_key,
                    family_name,
                    per_dataset_rtfx=(family_key in ("public", "extra")),
                )
        else:
            if presence_substr in all_dataset_ids:
                print_csv_block(header, col_map, family_key, family_name)

    return composite_wer, results
