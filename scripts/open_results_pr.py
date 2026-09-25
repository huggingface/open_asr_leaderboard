#!/usr/bin/env python3
"""Pull results from an HF bucket and open a PR against the published results CSV.

Scores the bucket exactly the way `scripts/score_bucket_results.py` does, turns the
CSV summary block into one row, upserts that row into the target dataset's CSV, and
opens a pull request on the Hub.

Nothing is pushed unless --open_pr or --merge is passed: the default is a dry run
that prints the row and the diff. --merge opens the PR and merges it straight away,
and is refused up front unless the token can write to every target repo.

Usage:
    # dry run (default): show the row that would be added
    python scripts/open_results_pr.py --target english --model_id nvidia/parakeet-tdt-0.6b-v3

    # the same, with the metadata columns the English sheet expects
    python scripts/open_results_pr.py --target english --model_id my-org/my-model \
        --license apache-2.0 --size_b 0.6 --num_languages 25 \
        --encoder FastConformer --decoder TDT \
        --training_data_disclosure https://huggingface.co/my-org/my-model#training-data \
        --open_pr

    # private sets (each lives in its own dataset repo)
    python scripts/open_results_pr.py --target appen      --model_id my-org/my-model
    python scripts/open_results_pr.py --target dataocean  --model_id my-org/my-model
    python scripts/open_results_pr.py --target voicearena --model_id my-org/my-model

    # public multilingual: one file per language
    python scripts/open_results_pr.py --target multilingual --language de --model_id my-org/my-model
    python scripts/open_results_pr.py --target multilingual --model_id my-org/my-model  # all languages

    # an API model: results are read from the private bucket for every target
    python scripts/open_results_pr.py --target english --model_id my-org/my-api-model --api

    # re-run a model and replace its row outright (blank what was not re-scored)
    python scripts/open_results_pr.py --target english --model_id my-org/my-model --overwrite

    # open the PR and merge it into main immediately (needs write access)
    python scripts/open_results_pr.py --target english --model_id my-org/my-model --merge

    # re-use results already synced locally
    python scripts/open_results_pr.py --target english --model_id my-org/my-model --skip_sync

    # every target the model has results for (syncs each bucket once)
    python scripts/open_results_pr.py --model_id my-org/my-model
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from huggingface_hub import HfApi, hf_hub_download

from normalizer.eval_utils import score_results
from score_bucket_results import ML_LANGUAGES, sync_bucket

# Buckets the results are read from, per target.
ENGLISH_BUCKET = "hf-audio/asr_leaderboard_h200"
PRIVATE_BUCKET = "hf-audio/asr_leaderboard_private"
MULTILINGUAL_BUCKET = "hf-audio/asr_leaderboard_multilingual"

# Language files that mark an API model's RTFx as -1 rather than leaving it blank
# (multilingual_hi.csv and the English sheet use blank).
RTFX_MINUS_ONE_LANGUAGES = {"de", "fr", "it", "es", "pt", "nl"}

# Tried in this order when --target is omitted.
ALL_TARGETS = ["english", "appen", "dataocean", "voicearena", "multilingual"]

# What --api writes into the License column when --license is not given: every
# one of the 14 API rows in english_short_latest.csv uses exactly this.
API_LICENSE = "Proprietary"

# Metadata columns that scoring cannot know; supplied by CLI flags and written
# into whichever target columns exist with these names.
METADATA_ARGS = {
    "License": "license",
    "Size (B)": "size_b",
    "# Languages": "num_languages",
    "Encoder": "encoder",
    "Decoder": "decoder",
    "Training data disclosure": "training_data_disclosure",
}


class Target:
    """One published results file and how to produce a row for it.

    passes: (families, language) pairs handed to score_results. Several are needed
        when a target's columns come from more than one family block, or when the
        blocks need different normalizers (voicearena mixes English and Hindi).
    renames: generated CSV label -> column name in the published file. No target
        needs this now that normalizer/eval_utils.py emits the published labels;
        kept as the escape hatch for the next time a sheet diverges.
    """

    def __init__(self, name, repo_id, filename, bucket, passes, renames=None,
                 avg_column=None, avg_sources=None, api_rtfx=""):
        self.name = name
        self.repo_id = repo_id
        self.filename = filename
        self.bucket = bucket
        self.passes = passes
        self.renames = renames or {}
        # avg_column is recomputed from avg_sources on the *merged* row. Without
        # this, re-running a single dataset would overwrite the published average
        # with the mean of just that dataset.
        self.avg_column = avg_column
        self.avg_sources = avg_sources or []
        # What --api writes into the RTFx columns. The published sheets disagree:
        # english_short_latest.csv and multilingual_hi.csv leave them blank for API
        # models, multilingual_{de,fr,it,es,pt,nl}.csv use -1. Each target keeps its
        # own file's convention rather than introducing a third.
        self.api_rtfx = api_rtfx


def build_targets(language=None):
    """Targets keyed by --target value. `language` selects the multilingual file."""
    targets = {
        # The English sheet spans two family blocks: `public` supplies the avg /
        # RTFx / per-dataset columns, `extra` the four non-cleaned WER columns.
        "english": Target(
            "english",
            "hf-audio/open-asr-leaderboard-results",
            "english_short_latest.csv",
            ENGLISH_BUCKET,
            passes=[(["public"], "en"), (["extra"], "en")],
            avg_column="avg",
            avg_sources=[
                "AMI-Cleaned WER",
                "Earnings22-Cleaned-AA-chunked WER",
                "Gigaspeech-Cleaned WER",
                "LS Clean WER",
                "LS Other WER",
                "SPGISpeech WER",
                "Voice Arena Monsoon WER",
                "Voxpopuli-AA-Cleaned WER",
            ],
        ),
        "dataocean": Target(
            "dataocean",
            "hf-audio/dataocean_shortform_results",
            "dataocean_short_latest.csv",
            PRIVATE_BUCKET,
            passes=[(["dataocean"], "en")],
        ),
        "appen": Target(
            "appen",
            "hf-audio/appen_shortform_results",
            "appen_short_latest.csv",
            PRIVATE_BUCKET,
            passes=[(["appen"], "en")],
        ),
        # One file, two family blocks, two normalizers: the Hindi set is scored
        # with voi_oiwer (see OIWER_LANGUAGES in normalizer/eval_utils.py).
        "voicearena": Target(
            "voicearena",
            "hf-audio/voicearena_shortform_results",
            "voicearena_short_latest.csv",
            PRIVATE_BUCKET,
            passes=[(["voicearena_private"], "en"), (["voicearena_private_hi"], "hi")],
        ),
    }
    if language:
        targets["multilingual"] = Target(
            f"multilingual-{language}",
            "hf-audio/multilingual_evals",
            f"multilingual_{language}.csv",
            MULTILINGUAL_BUCKET,
            passes=[([f"ml_{language}"], language)],
            # No multilingual_*.csv carries an `avg` column at present, so this is
            # inert; leaving the sources empty derives them from whatever WER
            # columns the file has, so it fills in if one ever gains the column.
            avg_column="avg",
            api_rtfx="-1" if language in RTFX_MINUS_ONE_LANGUAGES else "",
        )
    return targets


# Org roles that may push to an existing repo. "contributor" is left out: it can
# only write to repos the member created, which none of the results repos are.
WRITE_ORG_ROLES = {"write", "admin"}


def check_write_access(api, repo_id, repo_type, token):
    """Return (ok, reason) for whether `token` can push to `repo_id`.

    Decided from whoami() rather than by attempting a write, so --merge fails before
    any bucket is synced or PR opened. Two things must both hold: the token itself
    grants write (a "write" token, or a fine-grained token scoped with repo.write on
    the repo or its namespace), and the account has write rights in that namespace
    (it is the owning user, or holds a write/admin role in the owning org).
    """
    try:
        info = api.whoami(token=token)
    except Exception as exc:
        return False, f"could not authenticate ({exc})"

    namespace = repo_id.split("/")[0]
    user = info.get("name")
    if namespace == user:
        account_ok = True
    else:
        roles = {o.get("name"): o.get("roleInOrg") for o in info.get("orgs", [])}
        account_ok = roles.get(namespace) in WRITE_ORG_ROLES
        if not account_ok:
            return False, (
                f"{user} has role {roles.get(namespace) or 'none'!r} in {namespace}; "
                f"need one of {sorted(WRITE_ORG_ROLES)}"
            )

    token_info = info.get("auth", {}).get("accessToken", {})
    role = token_info.get("role")
    if role == "write":
        return True, f"{user} ({role} token)"
    if role == "fineGrained":
        scoped = (token_info.get("fineGrained") or {}).get("scoped", [])
        for entry in scoped:
            entity = entry.get("entity", {})
            covers = (
                entity.get("name") == repo_id and entity.get("type") == repo_type
            ) or (
                entity.get("name") == namespace and entity.get("type") in ("user", "org")
            )
            if covers and "repo.write" in entry.get("permissions", []):
                return True, f"{user} (fine-grained token, repo.write on {entity['name']})"
        return False, f"fine-grained token has no repo.write on {repo_id} or {namespace}"
    return False, f"token role is {role!r}; need a write or fine-grained token"


def parse_csv_block(text):
    """Pull {column: value} out of the CSV summary block score_results prints.

    The printed block is the canonical row format -- it is where avg, RTFx and the
    per-dataset ordering are worked out -- so it is read back rather than
    recomputed here, which would be a second implementation free to drift.
    """
    rows = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("model,"):
            continue
        header = next(csv.reader([line]))
        for data_line in lines[i + 1 :]:
            if data_line.startswith("*") or not data_line.strip():
                break
            values = next(csv.reader([data_line]))
            if len(values) != len(header):
                print(
                    f"WARNING: skipping a summary row with {len(values)} fields "
                    f"(header has {len(header)}): {data_line[:80]}",
                    file=sys.stderr,
                )
                continue
            rows[values[0]] = dict(zip(header, values))
    return rows


def score_one(local_dir, model_id, families, language):
    """Run score_results for one family group and return {model: {column: value}}."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            score_results(
                local_dir,
                model_id=model_id,
                csv_only=True,
                language=language,
                families=families,
            )
    except ValueError as exc:
        # No manifests for this family -- normal when a model was not run on it.
        print(f"  [{'+'.join(families)}] skipped: {exc}")
        return {}
    return parse_csv_block(buf.getvalue())


def collect_row(target, local_dir, model_id):
    """Merge every pass for `target` into one {column: value} row."""
    merged = {}
    for families, language in target.passes:
        rows = score_one(local_dir, model_id, families, language)
        if not rows:
            continue
        # score_results labels the row with the model id it was given; fall back
        # to the single row present when only one model was scored.
        row = rows.get(model_id) or (next(iter(rows.values())) if len(rows) == 1 else None)
        if row is None:
            print(f"  [{'+'.join(families)}] no row for {model_id}; found {list(rows)}")
            continue
        for column, value in row.items():
            if column == "model":
                continue
            renamed = target.renames.get(column, column)
            # Keep the first non-empty value: a later pass must not blank a
            # column an earlier one filled.
            if value != "" or renamed not in merged:
                merged[renamed] = value
        print(f"  [{'+'.join(families)}] {len(row) - 1} columns")
    return merged


def fetch_csv(repo_id, filename, hf_token):
    path = hf_hub_download(repo_id, filename, repo_type="dataset", token=hf_token)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def upsert(header, rows, model_id, values, metadata, sort, overwrite=False):
    """Insert or replace `model_id`'s row. Returns (rows, action, new_row, cleared).

    An existing row is always replaced in place, keeping its position in the file.
    What happens to a column this run produced no value for depends on `overwrite`:

    merge (default)
        Keep whatever the published row already has. Safe for scoring one dataset
        at a time and building the row up across several PRs, but a column left
        over from an earlier run stays, so one row can mix two runs.
    overwrite
        Blank it, so the row holds this run's results and nothing else. Metadata
        columns are still kept unless their flag was passed, since scoring can
        never produce them.
    """
    new_row = []
    for column in header:
        if column == "model":
            new_row.append(model_id)
        elif column in metadata and metadata[column] is not None:
            new_row.append(str(metadata[column]))
        else:
            new_row.append(values.get(column, ""))

    for i, row in enumerate(rows):
        if row and row[0].strip() == model_id:
            padded = row + [""] * (len(header) - len(row))
            merged, cleared = [], []
            for column, fresh, old in zip(header, new_row, padded):
                if fresh != "":
                    merged.append(fresh)
                elif not overwrite or column in METADATA_ARGS:
                    # Metadata is curated by hand and scoring cannot regenerate
                    # it, so --overwrite does not drop it either.
                    merged.append(old)
                else:
                    merged.append("")
                    if old.strip():
                        cleared.append((column, old))
            rows = rows[:i] + [merged] + rows[i + 1 :]
            return rows, "updated", merged, cleared

    rows = rows + [new_row]
    if sort:
        rows = sorted(rows, key=lambda r: (r[0] or "").lower())
    return rows, "added", new_row, []


def write_csv(header, rows):
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def process(target, args, api, synced):
    print(f"\n{'=' * 78}\n{target.repo_id}/{target.filename}\n{'=' * 78}")

    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")
    # Targets read from different buckets; sync each one once per run. `hf buckets
    # sync` defaults to --no-delete, so several buckets can share one directory.
    # API models are evaluated on the private infrastructure and every one of their
    # runs lands in PRIVATE_BUCKET, whatever sheet the row is destined for, so --api
    # overrides the target's own bucket. An explicit --bucket still wins.
    if args.bucket:
        bucket = args.bucket
    elif args.api and target.bucket != PRIVATE_BUCKET:
        bucket = PRIVATE_BUCKET
        print(f"--api: reading from {PRIVATE_BUCKET} instead of {target.bucket}")
    else:
        bucket = target.bucket
    if not args.skip_sync and bucket not in synced:
        sync_bucket(bucket, local_dir, hf_token=args.hf_token)
        synced.add(bucket)
    if not os.path.isdir(local_dir):
        print(f"ERROR: results directory not found: {local_dir}", file=sys.stderr)
        return False

    values = collect_row(target, local_dir, args.model_id)
    if not values:
        print(f"No scored columns for {args.model_id}; nothing to submit.")
        return False

    header, rows = fetch_csv(target.repo_id, target.filename, args.hf_token)
    metadata = {col: getattr(args, arg) for col, arg in METADATA_ARGS.items()}
    # Guarded on the column existing so --api does not warn about a License
    # column on the sheets that have none (appen / dataocean / multilingual).
    license_defaulted = (
        args.api and metadata.get("License") is None and "License" in header
    )
    if license_defaulted:
        metadata["License"] = API_LICENSE
    unknown = [c for c in values if c not in header]
    if unknown:
        print(f"WARNING: dropping columns absent from the published header: {unknown}")
    ignored = [c for c, v in metadata.items() if v is not None and c not in header]
    if ignored:
        print(f"WARNING: {target.filename} has no column(s) {ignored}; those flags are ignored.")

    rows, action, new_row, cleared = upsert(
        header, rows, args.model_id, values, metadata, args.sort, args.overwrite
    )
    if cleared:
        print(
            f"\n--overwrite: clearing {len(cleared)} column(s) this run did not "
            f"score, which held values from an earlier run:"
        )
        for column, old in cleared:
            print(f"  {column:<36} {old}  ->  (blank)")
    elif action == "updated" and not args.overwrite:
        stale = [
            c
            for c, v in zip(header, new_row)
            if v.strip() and c not in values and c not in METADATA_ARGS and c != "model"
            and c != target.avg_column
        ]
        if stale:
            print(
                f"\nNOTE: {len(stale)} column(s) kept from the published row because "
                f"this run produced no value for them: {', '.join(stale)}.\n"
                f"      Pass --overwrite to blank them instead."
            )

    if target.avg_column and target.avg_column in header:
        index = {c: i for i, c in enumerate(header)}
        # English must average only the eight cleaned sets, not the four extra
        # WER columns, so it lists them; elsewhere every WER column counts.
        avg_sources = target.avg_sources or [c for c in header if c.endswith(" WER")]
        present, missing = [], []
        for column in avg_sources:
            raw = new_row[index[column]] if column in index else ""
            (present if raw.strip() else missing).append(
                float(raw) if raw.strip() else column
            )
        if missing:
            print(
                f"WARNING: {len(missing)} of {len(avg_sources)} datasets have no "
                f"result ({', '.join(missing)}).\n"
                f"         '{target.avg_column}' is averaged over the {len(present)} present; "
                f"'RTFx' covers only the datasets this run scored.",
                file=sys.stderr,
            )
        if present:
            # Unrounded, matching english_short_latest.csv.
            new_row[index[target.avg_column]] = str(sum(present) / len(present))
            for i, row in enumerate(rows):
                if row and row[0].strip() == args.model_id:
                    rows[i] = new_row
                    break

    if args.api:
        placeholder = args.api_rtfx if args.api_rtfx is not None else target.api_rtfx
        index = {c: i for i, c in enumerate(header)}
        rtfx_columns = [c for c in header if c == "RTFx" or c.endswith(" RTFx")]
        for column in rtfx_columns:
            new_row[index[column]] = placeholder
        # Applied to the merged row, not just the scored values, so an earlier
        # upload's RTFx is cleared too rather than surviving the merge.
        for i, row in enumerate(rows):
            if row and row[0].strip() == args.model_id:
                rows[i] = new_row
                break
        shown = repr(placeholder) if placeholder else "blank"
        note = f"\n--api: {len(rtfx_columns)} RTFx column(s) set to {shown}."
        if license_defaulted:
            note += f" License set to {API_LICENSE!r} (pass --license to override)."
        print(note)

    print(f"\nRow ({action}):")
    for column, value in zip(header, new_row):
        if value != "":
            print(f"  {column:<36} {value}")
    blanks = [c for c, v in zip(header, new_row) if v == ""]
    if blanks:
        print(f"  (blank: {', '.join(blanks)})")

    content = write_csv(header, rows)
    if not (args.open_pr or args.merge):
        print("\nDry run - not opening a PR. Re-run with --open_pr (or --merge) to submit.")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(content)
            print(f"Wrote updated CSV to {args.out}")
        return True

    message = args.commit_message or f"Add {args.model_id} results"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write(content)
        tmp = fh.name
    try:
        pr = api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=target.filename,
            repo_id=target.repo_id,
            repo_type="dataset",
            token=args.hf_token,
            commit_message=message,
            create_pr=True,
        )
    finally:
        os.unlink(tmp)
    print(f"\nPR opened: {getattr(pr, 'pr_url', pr)}")
    if args.merge:
        # Merged through the PR rather than committed to main directly, so the
        # change keeps a discussion page on the Hub to link to and revert from.
        api.merge_pull_request(
            target.repo_id,
            pr.pr_num,
            token=args.hf_token,
            comment="Merged by open_results_pr.py --merge",
            repo_type="dataset",
        )
        print(f"PR #{pr.pr_num} merged into main.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Open a PR adding a model's results to the published CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        default=None,
        choices=ALL_TARGETS,
        help="Which published results file to update. Omit to try every target and "
             "submit to each one the model has results for.",
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="Model id as it should appear in the 'model' column, e.g. my-org/my-model.",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        choices=ML_LANGUAGES,
        metavar="LANGUAGE",
        help="For --target multilingual: language file(s) to update (repeatable). "
             f"Choices: {', '.join(ML_LANGUAGES)}. Defaults to all.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Override the source bucket. Takes precedence over the target's default "
             "and over the bucket --api selects.",
    )
    parser.add_argument("--local_dir", default=None, help="Where to sync results (default <repo>/results).")
    parser.add_argument("--skip_sync", action="store_true", help="Score already-downloaded results.")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN.")
    parser.add_argument("--open_pr", action="store_true", help="Actually open the PR (default: dry run).")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Open the PR and merge it into main immediately (implies --open_pr). "
             "Only allowed when the token can write to every target repo; checked "
             "before anything is synced or uploaded.",
    )
    parser.add_argument("--commit_message", default=None, help="PR title.")
    parser.add_argument("--out", default=None, help="Dry run: also write the updated CSV here.")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Model is served through an API. Reads results from "
             f"{PRIVATE_BUCKET} for every target, since that is where API runs land. "
             "Also does not upload RTFx: throughput measured over a network call is "
             "not comparable with a local GPU run, so the published sheets leave it "
             "out for API models. Clears every RTFx column, including any value "
             f"already in the row, and sets License to {API_LICENSE!r} unless "
             "--license is given.",
    )
    parser.add_argument(
        "--api_rtfx",
        default=None,
        metavar="VALUE",
        help="Override what --api writes into the RTFx columns. Defaults to the "
             "convention of the target file (blank for english/hi, -1 for the other "
             "multilingual languages).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing row outright: columns this run did not score are "
             "blanked instead of keeping the published value. Use when re-running a "
             "model so its row holds one run's results rather than a mix. Metadata "
             "columns are still preserved unless their flag is passed.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Re-sort rows case-insensitively by model when adding a new one "
             "(default: append, leaving existing order untouched).",
    )

    meta = parser.add_argument_group(
        "metadata columns",
        "Scoring cannot derive these; they are written into whichever target "
        "columns carry these names (all six on the English sheet, 'Size (B)' also "
        "on multilingual_hi.csv). Omitted flags leave the column as-is.",
    )
    meta.add_argument("--license", default=None, help="License column.")
    meta.add_argument("--size_b", default=None, help="Size (B) column, total parameters in billions.")
    meta.add_argument("--num_languages", default=None, help="# Languages column.")
    meta.add_argument("--encoder", default=None, help="Encoder column.")
    meta.add_argument("--decoder", default=None, help="Decoder column.")
    meta.add_argument(
        "--training_data_disclosure",
        default=None,
        help="Training data disclosure column (a URL to the model card section).",
    )
    args = parser.parse_args()

    if args.api_rtfx is not None and not args.api:
        print("WARNING: --api_rtfx has no effect without --api.", file=sys.stderr)

    if args.language and args.target is not None and args.target != "multilingual":
        print(
            f"WARNING: --language only applies to --target multilingual; ignoring "
            f"{' '.join(args.language)}.",
            file=sys.stderr,
        )

    api = HfApi()

    plan = []
    for key in [args.target] if args.target else ALL_TARGETS:
        if key == "multilingual":
            plan += [(key, language) for language in (args.language or ML_LANGUAGES)]
        else:
            plan.append((key, None))

    if args.merge:
        repos = sorted({build_targets(language)[key].repo_id for key, language in plan})
        denied = []
        for repo_id in repos:
            ok, reason = check_write_access(api, repo_id, "dataset", args.hf_token)
            print(f"--merge: {repo_id}: {'write access' if ok else 'DENIED'} - {reason}")
            if not ok:
                denied.append(repo_id)
        if denied:
            print(
                f"ERROR: --merge needs write access to {', '.join(denied)}. "
                f"Use --open_pr to open a PR for a maintainer to merge instead.",
                file=sys.stderr,
            )
            sys.exit(2)

    synced = set()
    outcomes = []
    for key, language in plan:
        target = build_targets(language)[key]
        outcomes.append((target.name, process(target, args, api, synced)))

    submitted = [name for name, done in outcomes if done]
    skipped = [name for name, done in outcomes if not done]
    if len(plan) > 1:
        print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
        verb = "merged" if args.merge else "submitted" if args.open_pr else "would submit"
        print(f"  {verb}: {', '.join(submitted) if submitted else '(none)'}")
        print(f"  no results: {', '.join(skipped) if skipped else '(none)'}")
    sys.exit(0 if submitted else 1)


if __name__ == "__main__":
    main()
