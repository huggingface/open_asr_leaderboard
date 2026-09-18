"""Fetch and verify the exact reviewed Orukeet evaluator, then invoke its launcher."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from huggingface_hub import snapshot_download

SPACE = "oruk/open-asr-leaderboard-orukeet"
REVISION = "171ab5e79454cc3772a8700479cb150cfe4cc84f"
BUNDLE_SHA256 = "0cb403d1553052f0396770be499558cdd2a0d88dd29e73481121b54f715af5cd"


def main():
    if any(arg.split("=", 1)[0] in {"--source-revision", "--space"} for arg in sys.argv[1:]):
        raise SystemExit("The evaluator Space and revision are pinned by this submission.")
    scratch = TemporaryDirectory(prefix="orukeet-open-asr-")
    directory = Path(snapshot_download(
        repo_id=SPACE, repo_type="space", revision=REVISION,
        allow_patterns=["evaluator/**"], local_dir=scratch.name,
    )) / "evaluator"
    manifest = directory / "bundle-manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != BUNDLE_SHA256:
        raise ValueError("Evaluator bundle manifest checksum differs")
    for name, expected in json.loads(manifest.read_text())["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Evaluator bundle path escapes its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Evaluator bundle file checksum differs: " + name)
    # The downloaded launcher defaults to a dry run; billed execution requires
    # --execute and a new --receipt-directory. Its token is a Job secret.
    # Put immutable arguments last so argparse abbreviations cannot override them.
    command = [sys.executable, str(directory / "submit_jobs.py"), *sys.argv[1:],
               "--source-revision", REVISION, "--space", SPACE]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
