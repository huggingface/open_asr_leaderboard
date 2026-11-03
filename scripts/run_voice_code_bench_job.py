"""Submit an HF Job and check its terminal status, including on older HF CLIs."""

import argparse
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, Volume, get_token


def run_job(args, *, api=None, sleep=time.sleep):
    api = api or HfApi()
    secrets = {}
    for name in args.secret:
        value = get_token() if name == "HF_TOKEN" else os.environ.get(name)
        if not value:
            raise ValueError(f"Missing secret: {name}")
        secrets[name] = value
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("A job command is required.")
    job = api.run_job(
        image=args.image, command=command, flavor=args.flavor,
        timeout=args.timeout, namespace=args.namespace or None,
        env=dict(item.split("=", 1) for item in args.env), secrets=secrets,
        volumes=[Volume(type="bucket", source=args.bucket, mount_path="/results")],
        labels={"benchmark": "voice-code-bench", "stage": args.stage},
    )
    print(f"{args.stage} job: {job.url}", flush=True)
    record = {"id": job.id, "url": job.url, "namespace": job.owner.name, "stage": args.stage}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    previous = None
    while True:
        info = api.inspect_job(job_id=job.id, namespace=job.owner.name)
        status = info.status.stage if info.status else "UNKNOWN"
        if status != previous:
            print(f"{args.stage}: {status}", flush=True)
            record["status"] = status
            args.record.write_text(json.dumps(record, indent=2) + "\n")
            previous = status
        if status == "COMPLETED":
            return
        if status in {"ERROR", "CANCELED", "DELETED"}:
            raise RuntimeError(f"{args.stage} job {status}: {job.url}: {info.status.message}")
        sleep(args.poll_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--timeout", default="8h")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--stage", choices=["inference", "grading"], required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--secret", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--poll-interval", type=float, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    run_job(args)


if __name__ == "__main__":
    main()
