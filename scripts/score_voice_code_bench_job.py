"""HF grading entrypoint: install pinned metric helpers and checkpoint its cache."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, snapshot_download

from normalizer import voice_code_bench as scorer

# Judge selection. The protocol's judge is the pinned BF16 checkpoint (normalizer/voice_code_bench_judge.py), and
# stays the default. VCB_JUDGE=fp8 grades with Qwen's FP8 variant instead, as an experiment to compare against it.
# The FP8 judge has its own verifier id: cache headers and request digests bind the id and the runtime, so its cache
# can never be mixed with the BF16 judge's, and its report says which judge produced it. The prompt, request
# construction, validation and vLLM settings are unchanged (the judge module is deliberately not edited: its digest
# covers its own bytes, and changing them would make existing BF16 caches unreplayable).
JUDGES = {
    "fp8": {
        "id": "qwen3_8_27b_fp8_full_and_segments_v1",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "revision": "c3a462cc3d38cf351f3b26cbd763469eeb74ef81",
    },
}


def select_judge():
    name = os.environ.get("VCB_JUDGE", "bf16")
    if name == "bf16":
        return
    if name not in JUDGES:
        raise SystemExit(f"VCB_JUDGE must be bf16 or fp8, not {name!r}.")
    from normalizer import voice_code_bench_judge as judge

    chosen = JUDGES[name]
    judge.VERIFIER_ID = chosen["id"]
    judge.MODEL = chosen["model"]
    judge.MODEL_REVISION = chosen["revision"]
    judge.RUNTIME = {**judge.RUNTIME, "model": chosen["model"], "revision": chosen["revision"],
                     "tokenizer_revision": chosen["revision"]}
    print(f"Judge: {chosen['model']} @ {chosen['revision'][:7]} ({chosen['id']})", flush=True)


def main():
    package = snapshot_download(
        scorer.DATASET_ID, repo_type="dataset", revision=scorer.DATASET_REVISION,
        allow_patterns=["pyproject.toml", "README.md", "LICENSE", "scripts/voice_code_bench/**"],
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", package], check=True)
    select_judge()
    from normalizer.voice_code_bench_judge import VerifierCache

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="vcb-cache-") as cache_dir:
        class CheckpointCache(VerifierCache):
            def __init__(self, path):
                self.destination = Path(path)
                local = Path(cache_dir) / "verifier-cache.json"
                if self.destination.exists():
                    shutil.copyfile(self.destination, local)
                super().__init__(local)

            def checkpoint(self):
                # Atomic replacement and the shared thread lock happen locally.
                # Publish a complete object through the API instead of relying
                # on atomic rename/locking support on a bucket mount.
                api.batch_bucket_files(
                    os.environ["RESULTS_BUCKET"],
                    add=[(self.path, self.destination.relative_to("/results").as_posix())],
                )

        scorer.main(cache_factory=CheckpointCache)


if __name__ == "__main__":
    main()
