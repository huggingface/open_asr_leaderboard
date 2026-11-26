#!/usr/bin/env python3
"""Summarize a graded VoiceCodeBench run: WER and CTEM, printed as Markdown.

Downloads the run's transcripts and CTEM report from the results bucket, then
  * computes WER from the raw transcripts with the same English normalizer and scorer as the other short-form sets
    (normalizer.eval_utils.score_manifest), also giving RTFx,
  * reads CTEM and the per-entity verdicts from the grading job's report (<manifest>.ctem.json),
  * prints both, and writes <manifest>.summary.json next to them.

With --publish-to <model folder>, a complete run's summary (and its transcripts, unless they are already in that
folder) is uploaded to that folder of the bucket (e.g. nvidia-parakeet-tdt-0.6b-v3/), where the other results of the
model live. Partial runs (smoke tests,
MAX_EVAL_SAMPLES) are never published, so they cannot replace a full run. Neither are the results of the FP8 judge
experiment (--judge fp8, default $VCB_JUDGE): the model folder holds the protocol's BF16 judge results only.

CTEM (higher is better) and WER (lower is better) are reported side by side but are separate metrics.

Usage:
    PYTHONPATH=. python scripts/summarize_voice_code_bench.py \
        --manifest-path runs/<run-id>/<model-folder>/MODEL_<model>_DATASET_besimple-ai-voice-code-bench_default_test.jsonl \
        --model-id nvidia/parakeet-tdt-0.6b-v3 --publish-to <model-folder>
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi

from normalizer import data_utils
from normalizer.eval_utils import format_rtfx, format_wer, markdown_table, read_manifest, score_manifest

DEFAULT_BUCKET = "hf-audio/asr_leaderboard_voice_code"
# File names of each judge's CTEM report, judge cache and summary (see vcb_submit_grading in scripts/submit_utils.sh).
JUDGE_FILES = {
    "bf16": {"report": ".ctem.json", "cache": ".qwen27b-verifier-cache.json", "summary": ".summary.json"},
    "fp8": {"report": ".ctem-fp8.json", "cache": ".qwen27b-fp8-verifier-cache.json", "summary": ".summary-fp8.json"},
}


def wer_scores(manifest):
    """WER (%) and RTFx of the manifest, as for the other sets.

    References that are empty after normalization are left out, as data_utils.prepare_data does for the other
    datasets (the VoiceCodeBench loader skips that filter so that raw text reaches the entity judge).
    """
    scored = [row for row in manifest if data_utils.is_target_text_in_range(data_utils.normalizer(row["text"]))]
    score = score_manifest(scored)
    score["recordings_scored"] = len(scored)
    score["recordings_excluded"] = len(manifest) - len(scored)
    return score


def by_entity_type(report):
    totals = defaultdict(lambda: {"total": 0, "correct": 0})
    for recording in report["entity_scores"]:
        for entity_type, counts in recording["by_type"].items():
            totals[entity_type]["total"] += counts["total"]
            totals[entity_type]["correct"] += counts["correct"]
    return {
        entity_type: {**counts, "ctem_percent": 100 * counts["correct"] / counts["total"]}
        for entity_type, counts in sorted(totals.items())
    }


def build_summary(manifest, report, *, model_id, run_id, bucket, manifest_path, judge="bf16"):
    entity_scores = report["entity_scores"]
    if len(entity_scores) != len(manifest) or {e["datapoint_id"] for e in entity_scores} != {
        row["audio_id"] for row in manifest
    }:
        raise ValueError("The CTEM report does not belong to this manifest: its recordings differ.")

    wer = wer_scores(manifest)
    recordings = [
        {
            "audio_id": e["datapoint_id"],
            "correct_entities": e["correct_token_count"],
            "total_entities": e["gold_token_count"],
            "task_success": bool(e["task_success"]),
            "missed": [
                {"type": m["type"], "role": m.get("role"), "canonical": m["canonical"]} for m in e["missed"]
            ],
        }
        for e in entity_scores
    ]
    stem = PurePosixPath(manifest_path).with_suffix("").as_posix()
    return {
        "model_id": model_id,
        "dataset": report["dataset"],
        "dataset_revision": report["dataset_revision"],
        "complete_test_split": report["complete_test_split"],
        "run_id": run_id,
        "ctem": {
            "percent": report["ctem_percent"],
            "higher_is_better": True,
            "correct_entities": report["correct_entities"],
            "total_entities": report["total_entities"],
            "recordings": report["recordings"],
            "task_success_percent": 100 * sum(r["task_success"] for r in recordings) / len(recordings),
            "judge": {k: report["verifier"][k] for k in ("id", "model", "config_digest")}
            | {"revision": report["verifier"]["runtime"].get("revision")},
        },
        "wer": {
            "percent": wer["wer"],
            "higher_is_better": False,
            "ins": wer["ins"],
            "del": wer["del"],
            "sub": wer["sub"],
            "recordings_scored": wer["recordings_scored"],
            "recordings_excluded": wer["recordings_excluded"],
            "normalizer": "normalizer.data_utils.normalizer (English), as for the other short-form sets",
        },
        "rtfx": wer["rtfx"],
        "by_entity_type": by_entity_type(report),
        "recordings": recordings,
        "artifacts": {
            "bucket": bucket,
            "manifest": manifest_path,
            "ctem_report": stem + JUDGE_FILES[judge]["report"],
            "judge_responses": stem + JUDGE_FILES[judge]["cache"],
        },
    }


def format_summary(summary):
    ctem, wer = summary["ctem"], summary["wer"]
    label = "full test split" if summary["complete_test_split"] else "PARTIAL smoke test"
    headline = markdown_table(
        ["Metric", "Value"],
        [
            ["CTEM (%, higher is better)", f"{ctem['percent']:.2f}"],
            ["Entities recovered", f"{ctem['correct_entities']}/{ctem['total_entities']}"],
            ["Recordings with every entity recovered (%)", f"{ctem['task_success_percent']:.2f}"],
            ["WER (%, lower is better)", format_wer(wer["percent"])],
            ["RTFx", format_rtfx(summary["rtfx"])],
        ],
    )
    types = markdown_table(
        ["Entity type", "Recovered", "Total", "CTEM (%)"],
        [
            [t, c["correct"], c["total"], f"{c['ctem_percent']:.2f}"]
            for t, c in summary["by_entity_type"].items()
        ],
    )
    note = ""
    if wer["recordings_excluded"]:
        note = f"\nWER leaves out {wer['recordings_excluded']} recording(s) whose reference is empty after normalization."
    return (
        f"**VoiceCodeBench** — `{summary['model_id']}` ({label}, {ctem['recordings']} recordings, "
        f"judge {ctem['judge']['model']})\n\n{headline}\n\n{types}{note}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", required=True,
                        help="Graded manifest, relative to the bucket (runs/<run-id>/<model-folder>/<manifest>.jsonl).")
    parser.add_argument("--bucket", default=os.environ.get("RESULTS_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--model-id", help="Defaults to the model folder name of the manifest path.")
    parser.add_argument("--run-id", help="Defaults to the run directory of the manifest path.")
    parser.add_argument("--judge", choices=sorted(JUDGE_FILES), default=os.environ.get("VCB_JUDGE", "bf16"),
                        help="Which judge's report to summarize (the FP8 judge is an experiment). Default: $VCB_JUDGE or bf16.")
    parser.add_argument("--output-dir", type=Path, help="Where the files are downloaded and the summary is written.")
    parser.add_argument("--publish-to",
                        help="Bucket folder (the model folder) to upload a complete run's transcripts and summary to.")
    args = parser.parse_args()

    remote = PurePosixPath(args.manifest_path)
    if remote.is_absolute() or ".." in remote.parts or remote.suffix != ".jsonl":
        raise SystemExit("Expected a relative bucket manifest path ending in .jsonl.")
    run_id = args.run_id or (remote.parts[1] if remote.parts[0] == "runs" and len(remote.parts) > 2 else None)
    model_id = args.model_id or remote.parent.name
    output_dir = args.output_dir or Path("nemo_asr/results/voice_code_bench") / remote.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_local = output_dir / remote.name
    report_remote = remote.with_suffix(JUDGE_FILES[args.judge]["report"])
    report_local = output_dir / report_remote.name
    api = HfApi()
    # Always fetch: a regrade replaces the report of the same manifest.
    api.download_bucket_files(
        args.bucket,
        [(remote.as_posix(), manifest_local), (report_remote.as_posix(), report_local)],
        raise_on_missing_files=True,
    )

    summary = build_summary(
        read_manifest(str(manifest_local)), json.loads(report_local.read_text(encoding="utf-8")),
        model_id=model_id, run_id=run_id, bucket=args.bucket, manifest_path=remote.as_posix(), judge=args.judge,
    )
    summary_local = output_dir / remote.with_suffix(JUDGE_FILES[args.judge]["summary"]).name
    summary_local.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(format_summary(summary))
    print(f"\nSummary saved at: {summary_local}")

    if not args.publish_to:
        return
    if args.judge != "bf16":
        print(f"Not publishing to {args.publish_to}/: only the protocol's BF16 judge results go to the model folder.")
        return
    if not summary["complete_test_split"]:
        print(f"Not publishing to {args.publish_to}/: partial runs never replace a full run's results.")
        return
    folder = args.publish_to.strip("/")
    uploads = [(summary_local, f"{folder}/{summary_local.name}")]
    if remote.parent.as_posix() != folder:
        # Transcripts of a run directory are copied to the model folder; ones already in it stay as they are.
        uploads.insert(0, (manifest_local, f"{folder}/{manifest_local.name}"))
    api.batch_bucket_files(args.bucket, add=uploads)
    print(f"Published to hf://buckets/{args.bucket}/{folder}/: " + ", ".join(Path(r).name for _, r in uploads))


if __name__ == "__main__":
    main()
