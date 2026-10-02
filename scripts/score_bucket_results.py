#!/usr/bin/env python3
"""Download all results from an HF bucket and print a CSV summary.

Usage:
    python scripts/score_bucket_results.py
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard
    python scripts/score_bucket_results.py --bucket bezzam/asr_leaderboard --local_dir results
    python scripts/score_bucket_results.py --skip_sync   # re-score already-downloaded results
    python scripts/score_bucket_results.py --family appen --family dataocean   # non-default families
    python scripts/score_bucket_results.py --family voicearena_private_hi  # Hindi private set (voi_oiwer)
    python scripts/score_bucket_results.py --family all  # every detected family

    # Long-form (earnings21, earnings22, CORAAL) results, as on the leaderboard's
    # Long-form tab. Defaults to the hf-audio/asr_leaderboard_longform bucket.
    python scripts/score_bucket_results.py --family longform

    # Each manifest's WER / RTFx is stored in a `.score.json` file next to it (the
    # long-form eval job writes it; otherwise the first scoring run does, and uploads
    # it to the bucket), so a manifest is aligned only once. After a change to the
    # normalizer, re-score everything and replace the stored scores:
    python scripts/score_bucket_results.py --family longform --recompute_scores

    # Multilingual (FLEURS/MCV/MLS) results. Defaults to the
    # hf-audio/asr_leaderboard_multilingual bucket, and scores each language
    # separately (each with its own normalizer).
    python scripts/score_bucket_results.py --multilingual
    python scripts/score_bucket_results.py --multilingual --language fr --language de
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict

# Allow importing normalizer from the repo root regardless of where the script
# is called from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from normalizer.eval_utils import SCORE_SUFFIX, score_results

# Languages covered by the multilingual (FLEURS/MCV/MLS + Hindi Monsoon) benchmarks.
ML_LANGUAGES = ["de", "fr", "it", "es", "pt", "nl", "hy", "hi"]

# Dataset families selectable via --family, and the language each is scored with.
# Families not listed in FAMILY_LANGUAGES are scored with the English normalizer;
# 'hi' routes through voi_oiwer (see OIWER_LANGUAGES in normalizer/eval_utils.py).
FAMILIES = ["appen", "dataocean", "public", "extra", "longform", "voicearena_private_hi", "voicearena_private"]
FAMILY_LANGUAGES = {"voicearena_private_hi": "hi"}

# Columns of the combined multilingual CSV summary: (column label, dataset substring).
ML_CSV_COLUMNS = [
    ("de_covost", "mcv_de_test"),
    ("de_fleurs", "fleurs_de_test"),
    ("fr_covost", "mcv_fr_test"),
    ("fr_mls", "mls_fr_test"),
    ("fr_fleurs", "fleurs_fr_test"),
    ("it_covost", "mcv_it_test"),
    ("it_mls", "mls_it_test"),
    ("it_fleurs", "fleurs_it_test"),
    ("es_covost", "mcv_es_test"),
    ("es_mls", "mls_es_test"),
    ("es_fleurs", "fleurs_es_test"),
    ("pt_mls", "mls_pt_test"),
    ("pt_fleurs", "fleurs_pt_test"),
    ("nl_covost", "mcv_nl_test"),
    ("nl_mls", "mls_nl_test"),
    ("nl_fleurs", "fleurs_nl_test"),
    ("hy_fleurs", "fleurs_hy_test"),
    ("hy_mcv", "mcv_hy_test"),
    ("hi_monsoon", "Monsoon_hi_test"),
]


def print_multilingual_csv(all_results: dict) -> None:
    """Print a combined CSV summary across all multilingual results.

    Columns: model, RTFx, <one column per language/dataset>, Avg.
    """
    # Group results per model.
    model_keys = sorted({key.split(" | ")[0].strip() for key in all_results})

    print()
    print("*" * 80)
    print("CSV Summary (multilingual):")
    print("*" * 80)
    header = ["model", "RTFx"] + [label for label, _ in ML_CSV_COLUMNS] + ["Avg"]
    print(",".join(header))

    for model_key in model_keys:
        model_results = {k: v for k, v in all_results.items() if k.split(" | ")[0].strip() == model_key}

        wer_vals = []
        for _label, ds_substr in ML_CSV_COLUMNS:
            wer = next(
                (v["wer"] for k, v in model_results.items() if ds_substr in k),
                None,
            )
            wer_vals.append(wer)

        # RTFx over all multilingual datasets with timing info.
        audio = sum(v["audio_length"] for v in model_results.values()
                    if v.get("audio_length") and v.get("inference_time"))
        time = sum(v["inference_time"] for v in model_results.values()
                   if v.get("audio_length") and v.get("inference_time"))
        rtfx = str(round(audio / time, 2)) if time else ""

        present = [v for v in wer_vals if v is not None]
        avg = str(round(sum(present) / len(present), 2)) if present else ""

        cols = [model_key, rtfx] + [str(v) if v is not None else "" for v in wer_vals] + [avg]
        print(",".join(cols))

    print("*" * 80)


def model_sync_patterns(model_id: str) -> list:
    """`hf buckets sync --include` patterns selecting one model's results.

    The same files score_results(model_id=...) picks out of a directory: anything in a
    folder named after the model, at any depth (some runs sit under an experiment
    folder, e.g. fast-gpu-asr-<date>/<model>/...), or whose name carries the model id
    (a few live in a folder named otherwise). Score files match too.
    """
    folder = model_id.replace("/", "-")
    return [f"{folder}/*", f"*/{folder}/*", f"*MODEL_{folder}_DATASET_*"]


def sync_bucket(bucket: str, local_dir: str, hf_token: str | None = None,
                model_ids: list | None = None) -> None:
    """Sync an HF bucket to a local directory using the `hf` CLI.

    model_ids: sync only these models' results (see model_sync_patterns) rather than
    the whole bucket.
    """
    bucket_url = f"hf://buckets/{bucket}"
    include = []
    for model_id in model_ids or []:
        for pattern in model_sync_patterns(model_id):
            include += ["--include", pattern]
    scope = f" ({', '.join(model_ids)})" if model_ids else ""
    print(f"Syncing {bucket_url}{scope}  \u2192  {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    subprocess.run(
        ["hf", "buckets", "sync", bucket_url, local_dir, *include],
        check=True,
        env=env,
    )
    print("Sync complete.\n")


def upload_scores(bucket: str, local_dir: str, score_paths: list, hf_token: str | None = None) -> None:
    """Upload score files written while scoring into `bucket`, next to their manifests.

    `local_dir` can hold several buckets' results (`hf buckets sync` does not
    delete), so a score file is uploaded only if its manifest is in `bucket` at the
    same path and with the same size as the local copy that was scored.
    """
    if not score_paths:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    remote = {
        f.path: f.size
        for f in api.list_bucket_tree(bucket, recursive=True)
        if getattr(f, "type", None) == "file"
    }
    add, skipped = [], []
    for path in sorted(set(score_paths)):
        rel = os.path.relpath(path, local_dir).replace(os.sep, "/")
        manifest = rel.removesuffix(SCORE_SUFFIX) + ".jsonl"
        local_manifest = os.path.join(local_dir, manifest)
        if remote.get(manifest) == os.path.getsize(local_manifest):
            add.append((path, rel))
        else:
            skipped.append(rel)
    if add:
        api.batch_bucket_files(bucket, add=add)
        print(f"Uploaded {len(add)} score file(s) to hf://buckets/{bucket}.")
    if skipped:
        print(
            f"Not uploading {len(skipped)} score file(s) whose manifest is not in "
            f"hf://buckets/{bucket} as scored: {', '.join(skipped)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Score all results from an HF bucket.")
    parser.add_argument(
        "--bucket",
        default=None,
        help="HF bucket name (without the hf://buckets/ prefix). Defaults to "
             "hf-audio/asr_leaderboard_multilingual if --multilingual is set, "
             "hf-audio/asr_leaderboard_longform if --family longform is the only "
             "family, otherwise hf-audio/asr_leaderboard_h200.",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Local directory to sync results into. Defaults to <repo_root>/results.",
    )
    parser.add_argument(
        "--skip_sync",
        action="store_true",
        help="Skip the bucket sync and score the already-downloaded results in --local_dir.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for private buckets. Defaults to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        choices=FAMILIES + ["all"],
        metavar="FAMILY",
        help="Dataset family to include in the CSV summary (can be repeated). "
             f"Choices: {', '.join(FAMILIES)}, all. Defaults to public. "
             "Families requiring a non-English normalizer (e.g. voicearena_private_hi) are "
             "scored in a separate pass. Ignored when --multilingual is set.",
    )
    parser.add_argument(
        "--model_id",
        action="append",
        default=None,
        metavar="MODEL_ID",
        help="Score only this model (can be repeated for multiple models). "
             "E.g. --model_id zoom/scribe_v1 --model_id assembly/universal-3-pro. "
             "Defaults to scoring all models.",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Score multilingual (FLEURS/MCV/MLS) results instead of the English "
             "public benchmarks. Scores each language separately, since each "
             "requires its own normalizer.",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        choices=ML_LANGUAGES,
        metavar="LANGUAGE",
        help=f"Language(s) to score (can be repeated). Choices: {', '.join(ML_LANGUAGES)}. "
             "Implies --multilingual. Defaults to all languages found.",
    )
    parser.add_argument(
        "--recompute_scores",
        action="store_true",
        help="Re-score every manifest instead of reading its stored .score.json, and "
             "replace the stored scores (locally and in the bucket). Use after a "
             "change to the normalizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Processes used to align the transcripts of a manifest that has no stored "
             "score. Defaults to every CPU available to this process; 1 aligns serially.",
    )
    parser.add_argument(
        "--no_upload_scores",
        action="store_true",
        help="Keep newly written .score.json files local instead of uploading them "
             "to the bucket.",
    )
    args = parser.parse_args()

    # --language only has meaning for the multilingual pass, so passing it on its
    # own implies --multilingual; otherwise it would silently score the English
    # benchmarks and ignore the language. Must precede the bucket default below,
    # which keys off args.multilingual.
    if args.language and not args.multilingual:
        print(f"--language {' '.join(args.language)} given: assuming --multilingual.\n")
        args.multilingual = True

    # The multilingual pass derives its families from the language list, so a
    # --family selection has nowhere to apply. Say so rather than dropping it.
    if args.family and args.multilingual:
        print(
            f"WARNING: ignoring --family {' '.join(args.family)}: families do not "
            "apply to the multilingual pass, which scores one family per language.",
            file=sys.stderr,
        )

    if args.bucket:
        bucket = args.bucket
    elif args.multilingual:
        bucket = "hf-audio/asr_leaderboard_multilingual"
    elif args.family == ["longform"]:
        bucket = "hf-audio/asr_leaderboard_longform"
    else:
        bucket = "hf-audio/asr_leaderboard_h200"
    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")

    if not args.skip_sync:
        # With --model_id, only those models' results are downloaded.
        sync_bucket(bucket, local_dir, hf_token=args.hf_token, model_ids=args.model_id)
    else:
        print(f"Skipping sync — scoring results in: {local_dir}\n")

    if not os.path.isdir(local_dir):
        print(f"ERROR: Local results directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    # Score the requested models; csv_only=True suppresses per-dataset and
    # composite output, printing only the CSV summary block.
    model_ids = args.model_id or [None]  # None means all models
    written_scores = []
    cache_kwargs = dict(
        use_cache=True,
        recompute=args.recompute_scores,
        written_scores=written_scores,
        num_workers=args.num_workers,
    )

    if args.multilingual:
        languages = args.language or ML_LANGUAGES
        all_results = {}
        for model_id in model_ids:
            for language in languages:
                try:
                    _, results = score_results(
                        local_dir,
                        model_id=model_id,
                        multilingual=True,
                        language=language,
                        families=[f"ml_{language}"],
                        csv_only=True,
                        **cache_kwargs,
                    )
                    all_results.update(results)
                except ValueError as e:
                    print(f"Skipping language={language} model_id={model_id}: {e}")
        if all_results:
            print_multilingual_csv(all_results)
    else:
        families = args.family or ["public"]
        if "all" in families:
            families = FAMILIES

        # Each score_results call applies a single normalizer, so families are
        # grouped by the language they must be scored with and passed one group
        # per call.
        families_by_language = defaultdict(list)
        for family in families:
            families_by_language[FAMILY_LANGUAGES.get(family, "en")].append(family)

        for model_id in model_ids:
            for language, language_families in families_by_language.items():
                try:
                    score_results(
                        local_dir,
                        model_id=model_id,
                        csv_only=True,
                        language=language,
                        families=language_families,
                        **cache_kwargs,
                    )
                except ValueError as e:
                    print(f"Skipping families={language_families} model_id={model_id}: {e}")

    if not args.no_upload_scores:
        upload_scores(bucket, local_dir, written_scores, hf_token=args.hf_token)


if __name__ == "__main__":
    main()
