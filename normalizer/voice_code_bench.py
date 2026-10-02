"""VoiceCodeBench loading and CTEM scoring using its versioned official verifier."""

import argparse
import json
from pathlib import Path

DATASET_ID = "besimple-ai/voice-code-bench"
DATASET_REVISION = "bef2824f83ef1c796f3e79731a3b0741708730df"
VERIFIER_ID = "openai_gpt_5_5_v1"
MANIFEST_KEYS = ["audio_id", "voice_code_bench_revision"]


def is_voice_code_bench(dataset_path):
    return str(dataset_path).lower() == DATASET_ID


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


def score_manifest(manifest, metadata, *, cache, cache_mode="replay", allow_partial=False):
    from voice_code_bench.metrics import score_entity_capture
    from voice_code_bench.verifier import load_verifier_config, verify_entity_matches

    targets = {row["audio_id"]: row for row in metadata}
    if len(targets) != len(metadata) or not targets:
        raise ValueError("Expected non-empty metadata with unique audio IDs.")
    seen = set()
    # Validate the entire manifest before making any paid verifier requests.
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

    config = load_verifier_config(VERIFIER_ID)
    details = []
    for row in manifest:
        datapoint = targets[row["audio_id"]]
        # Keep punctuation, case, digits and spacing. WER normalization can
        # destroy the exact values CTEM is intended to measure.
        matches = verify_entity_matches(
            datapoint, row["pred_text"], config, {}, cache=cache, cache_mode=cache_mode
        )
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
            "id": config.id,
            "model": config.model,
            "config_digest": config.digest,
            "cache_mode": cache_mode,
        },
        "entity_scores": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Score raw VoiceCodeBench predictions with CTEM (higher is better).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to <manifest>.ctem.json.")
    parser.add_argument("--verifier-cache", type=Path, required=True)
    parser.add_argument("--verifier-mode", choices=["replay", "live-fill"], default="replay",
                        help="live-fill makes paid verifier calls on cache misses using OPENAI_API_KEY.")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from voice_code_bench.io import read_metadata
    from voice_code_bench.verifier import VerifierCache
    from .eval_utils import read_manifest

    metadata_path = hf_hub_download(
        DATASET_ID, "data/metadata.jsonl", repo_type="dataset", revision=DATASET_REVISION
    )
    report = score_manifest(
        read_manifest(args.manifest), read_metadata(Path(metadata_path)),
        cache=VerifierCache(args.verifier_cache), cache_mode=args.verifier_mode,
        allow_partial=args.allow_partial,
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
