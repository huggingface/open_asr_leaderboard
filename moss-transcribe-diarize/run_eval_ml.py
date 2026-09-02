"""Multilingual entry point for the MOSS Open ASR evaluator.

The inference implementation stays in ``run_eval.py`` so English and
multilingual evaluations cannot drift apart.
"""

from __future__ import annotations

import argparse

from run_eval import main as run_eval_main


def main(args: argparse.Namespace) -> int:
    """Adapt multilingual dataset arguments to the shared evaluator."""
    config_language = args.config_name.rsplit("_", 1)[-1]
    if args.language is not None and args.language != config_language:
        raise ValueError(
            f"Language {args.language!r} does not match config {args.config_name!r}"
        )

    args.dataset_path = args.dataset
    args.dataset = args.config_name
    return run_eval_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--model_revision",
        default="e5118b411bf5a77d7a90c4941066bec93c967312",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_max_new_tokens", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1)
    raise SystemExit(main(parser.parse_args()))
