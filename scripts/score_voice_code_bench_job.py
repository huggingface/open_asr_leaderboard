"""HF grading entrypoint: install the pinned verifier and checkpoint its cache."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, snapshot_download

from normalizer import voice_code_bench as scorer


def main():
    package = snapshot_download(
        scorer.DATASET_ID, repo_type="dataset", revision=scorer.DATASET_REVISION,
        allow_patterns=["pyproject.toml", "README.md", "LICENSE", "scripts/voice_code_bench/**"],
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", package], check=True)
    from voice_code_bench.verifier import VerifierCache

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="vcb-cache-") as cache_dir:
        class CheckpointCache(VerifierCache):
            def __init__(self, path):
                self.destination = Path(path)
                local = Path(cache_dir) / "verifier-cache.json"
                if self.destination.exists():
                    shutil.copyfile(self.destination, local)
                super().__init__(local)

            def put(self, entry):
                super().put(entry)
                # Atomic cache replacement and file locking happen on local disk.
                # Publish a complete object through the API instead of relying
                # on atomic rename/locking support on a bucket mount.
                api.batch_bucket_files(
                    os.environ["RESULTS_BUCKET"],
                    add=[(self.path, self.destination.relative_to("/results").as_posix())],
                )

        scorer.main(cache_factory=CheckpointCache)


if __name__ == "__main__":
    main()
