#!/usr/bin/env python3
# Adapted from the reference implementation accompanying "Quantifying Benchmark
# Optimization in ASR Models" (https://github.com/tlebryk/asr-benchmark-optimization,
# Apache-2.0).
"""Print the individual flagged spans `score_ref_rendering.py` scores, for inspection.

Each is shown with the untouched reference and predictions, extracted raw spans,
and agreement verdicts.

Models are joined on the reference text, so only clips every selected model
transcribed are shown; manifests keyed `sample_<i>` are therefore usable too.

Usage:
    python benchmark_fitting/show_flagged_spans.py --preds_dir results --dataset voxpopuli_test
    python benchmark_fitting/show_flagged_spans.py --preds_dir results --dataset ami_test \
        --models model-a,model-b --class spelling --limit 20 --html spans.html
"""

from __future__ import annotations

import argparse
import html
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from normalizer import to_hub_ids  # noqa: E402  (needs REPO_ROOT on sys.path)
from ref_rendering_utils import REPORTED_CLASSES, flagged_spans_for, score_clip
from score_ref_rendering import find_manifests, read_manifest

CONTEXT_WORDS = 6

CLASS_COLOR = {
    "spelling": "#d35400",
    "initialism": "#c0392b",
    "number": "#2471a3",
    "lexical": "#6c3483",
}


def load_hypotheses(manifests: dict[str, str], models: list[str]) -> dict[str, dict[str, tuple[str, str]]]:
    """Model to (reference join key -> (untouched reference, untouched hypothesis))."""
    out = {}
    for model in models:
        by_ref = {}
        for row in read_manifest(manifests[model]):
            if "text" not in row or "pred_text" not in row:
                continue
            by_ref[" ".join(row["text"].split())] = (row["text"], row["pred_text"])
        out[model] = by_ref
    return out


def collect_flagged_spans(hypotheses: dict[str, dict[str, str]], models: list[str], cls: str | None):
    """Every scored flagged span on the clips all selected models transcribed."""
    refs = set.intersection(*(set(hypotheses[m]) for m in models))
    out = []
    for ref_key in sorted(refs):
        raw_ref = hypotheses[models[0]][ref_key][0]
        flagged_spans, ref_full = flagged_spans_for(" ".join(raw_ref.split()))
        for index, fspan in enumerate(flagged_spans):
            if cls and fspan.cls != cls:
                continue
            out.append((ref_key, raw_ref, tuple(flagged_spans), tuple(ref_full), index))
    return out


def context(ref: str, raw_lo: int, raw_hi: int) -> tuple[str, str, str]:
    """``(left, span, right)`` with up to CONTEXT_WORDS words either side."""
    toks = ref.split()
    left = " ".join(toks[max(0, raw_lo - CONTEXT_WORDS) : raw_lo])
    right = " ".join(toks[raw_hi + 1 : raw_hi + 1 + CONTEXT_WORDS])
    if raw_lo - CONTEXT_WORDS > 0:
        left = "… " + left
    if raw_hi + 1 + CONTEXT_WORDS < len(toks):
        right = right + " …"
    return left, " ".join(toks[raw_lo : raw_hi + 1]), right


def verdicts(ref_key: str, flagged_spans, ref_full, index: int, hypotheses, models):
    """Raw prediction, extracted span, and verdict per model at one flagged span."""
    rows = []
    for model in models:
        _, raw_hyp = hypotheses[model][ref_key]
        _, eligible, agreed, hyp_raw = score_clip(list(flagged_spans), list(ref_full), raw_hyp)[index]
        if not eligible:
            rows.append((model, raw_hyp, hyp_raw, "not eligible"))
        else:
            rows.append((model, raw_hyp, hyp_raw, "agree" if agreed else "own"))
    return rows


def render_text(dataset, models, samples, hypotheses, total) -> str:
    hub = to_hub_ids(models)
    width = max(len(h) for h in hub.values())
    lines = [f"{dataset}: {len(samples)} of {total} flagged spans, {len(models)} models"]
    for n, (ref_key, raw_ref, flagged_spans, ref_full, index) in enumerate(samples, 1):
        candidate = flagged_spans[index]
        raw_span, norm_span, cls = candidate.raw, candidate.norm, candidate.cls
        raw_lo, raw_hi = candidate.raw_lo, candidate.raw_hi
        left, span, right = context(" ".join(raw_ref.split()), raw_lo, raw_hi)
        lines += [
            "",
            f"[{n}] {cls}",
            f"  context           {left} «{span}» {right}",
            f"  full raw ref      {raw_ref!r}",
            f"  reference span    {raw_span!r}",
            f"  normalizer        {norm_span!r}",
        ]
        for model, raw_hyp, hyp_raw, verdict in verdicts(
            ref_key, flagged_spans, ref_full, index, hypotheses, models
        ):
            lines.append(f"    {hub[model]:{width}}  {verdict:12} span={hyp_raw!r}")
            lines.append(f"    {'':{width}}  {'':12} full raw prediction={raw_hyp!r}")
    return "\n".join(lines) + "\n"


CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1c2833}
h1{font-size:20px} p.lede{color:#424949}
.fspan{margin:22px 0;padding:12px 14px;border:1px solid #d5dbdb;border-radius:8px}
.cls{font-size:11px;text-transform:uppercase;letter-spacing:.4px;font-weight:600}
.ctx{color:#566573}
.ctx mark{background:none;border-bottom:2px solid;padding:0 1px;font-weight:600;color:#1c2833}
table{border-collapse:collapse;font-size:13px;margin-top:10px;width:100%}
td,th{border:1px solid #e5e8e8;padding:4px 9px;text-align:left}
th{background:#fbfcfc;font-weight:600}
td.agree{background:#eef2f7} td.own{background:#f7f4ee} td.na{background:#fbfcfc;color:#b3b6b7}
td.v{white-space:nowrap;font-size:11px;text-transform:uppercase;letter-spacing:.3px}
td.full,.raw{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
code{font-size:12px}
.legend span{margin-right:18px;font-size:13px}
.legend .sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}
"""


def render_html(dataset, models, samples, hypotheses, total) -> str:
    hub = to_hub_ids(models)
    blocks = [
        f"<h1>Flagged spans — {html.escape(dataset)}</h1>",
        f'<p class="lede">{len(samples)} of {total} scored flagged spans, {len(models)} models. A flagged span is a span '
        "of the raw reference belonging to a curated rendering family erased by English WER normalization. "
        "Where a model reproduced the flagged span's words, its raw rendering is compared to the reference's.</p>",
        '<p class="legend">'
        '<span><span class="sw" style="background:#eef2f7"></span>same rendering as the reference</span>'
        '<span><span class="sw" style="background:#f7f4ee"></span>same words, different rendering</span>'
        '<span><span class="sw" style="background:#fbfcfc;border:1px solid #e5e8e8"></span>'
        "not eligible (words not reproduced)</span></p>",
    ]
    for n, (ref_key, raw_ref, flagged_spans, ref_full, index) in enumerate(samples, 1):
        candidate = flagged_spans[index]
        raw_span, norm_span, cls = candidate.raw, candidate.norm, candidate.cls
        raw_lo, raw_hi = candidate.raw_lo, candidate.raw_hi
        left, span, right = context(" ".join(raw_ref.split()), raw_lo, raw_hi)
        color = CLASS_COLOR.get(cls, "#7a869a")
        rows = []
        for model, raw_hyp, hyp_raw, verdict in verdicts(
            ref_key, flagged_spans, ref_full, index, hypotheses, models
        ):
            klass = {"agree": "agree", "own": "own"}.get(verdict, "na")
            shown = f"<code>{html.escape(hyp_raw)}</code>" if hyp_raw else "&mdash;"
            label = {"agree": "= reference", "own": "own"}.get(verdict, "not eligible")
            rows.append(
                f'<tr><th>{html.escape(hub[model])}</th><td class="{klass}">{shown}</td>'
                f'<td class="v {klass}">{label}</td>'
                f'<td class="full">{html.escape(raw_hyp)}</td></tr>'
            )
        blocks.append(
            f'<div class="fspan"><span class="cls" style="color:{color}">[{n}] {cls}</span>'
            f'<p class="ctx">{html.escape(left)} <mark style="border-color:{color}">{html.escape(span)}</mark> '
            f"{html.escape(right)}</p>"
            f'<p><strong>full raw reference</strong></p><div class="raw">{html.escape(raw_ref)}</div>'
            f"<p>reference span <code>{html.escape(raw_span)}</code> &rarr; normalizer "
            f'<code>{html.escape(norm_span) or "&empty;"}</code></p>'
            f'<table><tr><th>model</th><th>extracted raw span</th><th></th>'
            f'<th>full raw prediction</th></tr>{"".join(rows)}</table></div>'
        )
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>flagged spans — {html.escape(dataset)}</title>"
        f"<style>{CSS}</style>" + "".join(blocks)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preds_dir", required=True, help="Directory of prediction manifests.")
    parser.add_argument("--dataset", required=True, help="Dataset tag, e.g. voxpopuli_test.")
    parser.add_argument("--models", default=None, help="Comma-separated model ids. Default: every model found.")
    parser.add_argument(
        "--class",
        dest="cls",
        default=None,
        choices=list(REPORTED_CLASSES),
        help="Show only flagged spans of this class. Default: all classes.",
    )
    parser.add_argument("--limit", type=int, default=15, help="Number of flagged spans to sample. Default: 15")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed. Default: 0")
    parser.add_argument("--html", default=None, help="Write HTML here instead of printing text.")
    args = parser.parse_args()

    manifests = find_manifests(args.preds_dir, args.dataset)
    if not manifests:
        sys.exit(f"No manifests for {args.dataset} under {args.preds_dir}")
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in models if m not in manifests]
        if missing:
            sys.exit(f"No {args.dataset} manifest for: {', '.join(missing)}")
    else:
        models = sorted(manifests)

    hypotheses = load_hypotheses(manifests, models)
    flagged_spans = collect_flagged_spans(hypotheses, models, args.cls)
    if not flagged_spans:
        sys.exit("No flagged spans matched.")
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(range(len(flagged_spans)), min(args.limit, len(flagged_spans))))
    samples = [flagged_spans[i] for i in picked]

    if args.html:
        out = render_html(args.dataset, models, samples, hypotheses, len(flagged_spans))
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.html} ({len(samples)} of {len(flagged_spans)} flagged spans)")
    else:
        sys.stdout.write(render_text(args.dataset, models, samples, hypotheses, len(flagged_spans)))


if __name__ == "__main__":
    main()
