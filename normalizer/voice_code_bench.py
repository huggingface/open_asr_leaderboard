"""VoiceCodeBench loading and CTEM scoring using the pinned open-weight Qwen judge."""

import argparse
import json
from pathlib import Path

DATASET_ID = "besimple-ai/voice-code-bench"
DATASET_REVISION = "bef2824f83ef1c796f3e79731a3b0741708730df"
MANIFEST_KEYS = ["audio_id", "voice_code_bench_revision"]


def is_voice_code_bench(dataset_path):
    return str(dataset_path).lower() == DATASET_ID


def load_metadata(path):
    from voice_code_bench.io import read_metadata

    # This pinned release predates alternative_spellings. It defines only its
    # canonical spelling, so explicitly supply the tested judge's empty list.
    return [{**point, "entities": [{**entity, "alternative_spellings": []}
                                  for entity in point["entities"]]}
            for point in read_metadata(path)]


def load_data(args):
    from datasets import Audio, load_dataset
    from huggingface_hub import snapshot_download

    if args.split != "test" or args.dataset not in ("", "default"):
        raise ValueError("VoiceCodeBench supports only --dataset default --split test.")
    if args.streaming:
        raise ValueError("VoiceCodeBench currently requires downloading its audio; omit --streaming.")

    root = Path(snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=["data/metadata.jsonl", "data/audio/*.wav"],
    ))
    # The Hub dataset is JSON metadata, with file_name relative to data/.
    # It does not expose an Audio feature, so attach the WAVs explicitly.
    dataset = load_dataset(
        "json", data_files={"test": str(root / "data/metadata.jsonl")}, split="test"
    )

    def attach_audio(sample):
        audio_path = root / "data" / sample["file_name"]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        return {
            "audio": str(audio_path),
            "id": sample["audio_id"],
            "text": sample["transcripts"]["acoustic"],
            "voice_code_bench_revision": DATASET_REVISION,
        }

    return dataset.map(attach_audio, load_from_cache_file=False).cast_column("audio", Audio())


def score_manifest(manifest, metadata, *, cache, cache_mode="replay", allow_partial=False, workers=16):
    from voice_code_bench.metrics import score_entity_capture
    from .voice_code_bench_judge import VERIFIER_ID, MODEL, RUNTIME, config_digest, verify_manifest

    targets = {row["audio_id"]: row for row in metadata}
    if len(targets) != len(metadata) or not targets:
        raise ValueError("Expected non-empty metadata with unique audio IDs.")
    seen = set()
    # Validate the entire manifest before starting the GPU runtime.
    for row in manifest:
        audio_id = row.get("audio_id")
        if audio_id not in targets or audio_id in seen:
            raise ValueError(f"Unknown or duplicate audio_id: {audio_id!r}")
        seen.add(audio_id)
        if row.get("voice_code_bench_revision") != DATASET_REVISION:
            raise ValueError(f"Missing or incompatible VoiceCodeBench revision for {audio_id}.")
        if not isinstance(row.get("pred_text"), str):
            raise ValueError(f"Expected a raw pred_text string for {audio_id}.")
    if not seen or (not allow_partial and seen != set(targets)):
        raise ValueError("Expected the complete test split; use --allow-partial only for smoke tests.")

    # WER normalization must never change raw values before entity judging.
    judgments = verify_manifest(manifest, targets, cache=cache, cache_mode=cache_mode, workers=workers)
    details = []
    for row, matches in zip(manifest, judgments, strict=True):
        datapoint = targets[row["audio_id"]]
        prediction = {
            "audio_id": row["audio_id"],
            "model_transcript": row["pred_text"],
            "entity_matches": matches,
        }
        details.append(score_entity_capture(datapoint, "manifest", prediction))

    total = sum(row["gold_token_count"] for row in details)
    correct = sum(row["correct_token_count"] for row in details)
    if total == 0:
        raise ValueError("CTEM requires at least one target entity.")
    return {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "metric": "CTEM",
        "higher_is_better": True,
        "ctem_percent": 100 * correct / total,
        "correct_entities": correct,
        "total_entities": total,
        "recordings": len(details),
        "complete_test_split": seen == set(targets),
        "verifier": {
            "id": VERIFIER_ID,
            "model": MODEL,
            "config_digest": config_digest(),
            "runtime": RUNTIME,
            "transcript_layout": "full_and_segments",
            "cache_mode": cache_mode,
        },
        "entity_scores": details,
    }


def main(*, cache_factory=None):
    # The HF worker supplies a cache that checkpoints to the results bucket.
    # Standalone scoring uses the same Qwen cache on local disk.
    parser = argparse.ArgumentParser(description="Score raw VoiceCodeBench predictions with CTEM (higher is better).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to <manifest>.ctem.json.")
    parser.add_argument("--verifier-cache", type=Path, required=True)
    parser.add_argument("--verifier-mode", choices=["replay", "live-fill"], default="replay",
                        help="live-fill starts the pinned local GPU model for cache misses.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verifier-workers", type=int, default=16, help="Maximum simultaneous judge requests.")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from .voice_code_bench_judge import VerifierCache
    from .eval_utils import read_manifest

    metadata_path = hf_hub_download(
        DATASET_ID, "data/metadata.jsonl", repo_type="dataset", revision=DATASET_REVISION
    )
    report = score_manifest(
        read_manifest(args.manifest), load_metadata(Path(metadata_path)),
        cache=(cache_factory or VerifierCache)(args.verifier_cache), cache_mode=args.verifier_mode,
        allow_partial=args.allow_partial, workers=args.verifier_workers,
    )
    report["source_manifest"] = str(args.manifest)
    output = args.output or args.manifest.with_suffix(".ctem.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    label = "full test split" if report["complete_test_split"] else "PARTIAL smoke test"
    print(f"CTEM: {report['ctem_percent']:.2f}% ({report['correct_entities']}/{report['total_entities']} entities; {label})")
    print(f"Scores saved at: {output}")


if __name__ == "__main__":
    main()
