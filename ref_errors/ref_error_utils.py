# Copyright 2026 The Open ASR Leaderboard contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from the reference implementation accompanying "Quantifying Benchmark
# Optimization in ASR Models" (https://github.com/tlebryk/asr-benchmark-optimization,
# Apache-2.0), modules `align.py` and `refdis.py`. The alignment helpers are
# carried over verbatim; the edit finder is specialised to the case where the
# corrected transcript comes from a single human-corrected reference rather than
# from a panel of models.
"""Reference error agreement: alignment helpers and per-model scoring.

Where a benchmark's official reference disagrees with the audio, a model
transcribing the audio produces the audio's version. This module locates those
disagreements by diffing the official reference against a human-corrected one,
then records, for each model and each disagreement, which of the two the model's
output matches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher

__all__ = [
    "DEFAULTS",
    "RefEdit",
    "ref_error_agreement",
    "cer",
    "find_ref_errors",
    "insertions_by_anchor",
    "matched_count",
    "missed_indices",
    "substitution_at",
    "wilson",
]

DEFAULTS = {
    # Shortest edit worth counting, in reference tokens.
    "min_run_len": 1,
    # Count edits interior to the reference, not only those at either end.
    "include_middle": True,
    # A model must reproduce this share of the reference to be scored on the
    # clip at all. Without it, empty and off-language outputs dominate.
    "min_ref_match": 0.5,
    # A deletion edit only counts if what the correction puts in place of the
    # reference span is character-wise far from it. Below this threshold the
    # difference is spacing or accents, i.e. a normalization artifact.
    "min_consensus_cer": 0.30,
}


# ---------------------------------------------------------------------------
# Word-level alignment
#
# `difflib.SequenceMatcher` rather than a WER-style Levenshtein alignment:
# deterministic, dependency-free, and its opcodes (equal / replace / delete /
# insert) map onto the distinctions needed here. `autojunk` is off throughout —
# it drops frequent tokens, which here means the function words that carry much
# of the signal.
# ---------------------------------------------------------------------------


def _opcodes(ref_tokens: list[str], hyp: str):
    return SequenceMatcher(None, ref_tokens, (hyp or "").split(), autojunk=False).get_opcodes()


def missed_indices(ref_tokens: list[str], hyp: str) -> set[int]:
    """Reference positions the hypothesis did not reproduce.

    A position counts as missed under ``delete`` (the hypothesis skipped it) and
    under ``replace`` (the hypothesis put something else there).
    """
    out: set[int] = set()
    for tag, i1, i2, _j1, _j2 in _opcodes(ref_tokens, hyp):
        if tag in ("delete", "replace"):
            out.update(range(i1, i2))
    return out


def matched_count(ref_tokens: list[str], hyp: str) -> int:
    """Number of reference positions the hypothesis reproduced exactly.

    A competence gate: an empty, truncated or off-language hypothesis agrees
    with any span by accident.
    """
    return sum(i2 - i1 for tag, i1, i2, _j1, _j2 in _opcodes(ref_tokens, hyp) if tag == "equal")


def substitution_at(ref_tokens: list[str], hyp: str, span: set[int]) -> str:
    """The hypothesis text standing in for reference positions ``span``.

    Only ``replace`` opcodes contribute; a pure ``delete`` contributes the empty
    string, which is the correct reading — nothing was emitted there.
    """
    hyp_tokens = (hyp or "").split()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in _opcodes(ref_tokens, hyp):
        if tag == "replace" and span & set(range(i1, i2)):
            out.extend(hyp_tokens[j1:j2])
    return " ".join(out)


def insertions_by_anchor(ref_tokens: list[str], hyp: str) -> dict[int, list[str]]:
    """Map each reference boundary to the hypothesis tokens inserted there.

    Anchor ``0`` is before the first reference token; anchor ``len(ref_tokens)``
    is after the last. ``replace`` opcodes are excluded: their hypothesis tokens
    are substitutions, already visible through :func:`missed_indices`.
    """
    hyp_tokens = (hyp or "").split()
    out: dict[int, list[str]] = {}
    for tag, i1, _i2, j1, j2 in _opcodes(ref_tokens, hyp):
        if tag == "insert":
            out.setdefault(i1, []).extend(hyp_tokens[j1:j2])
    return out


def emitted_insertion(model_insert: list[str], target: list[str], at_start: bool) -> bool:
    """Did the model insert ``target`` at the same boundary alignment?

    Boundary-anchored, so ``target`` must be a suffix of the model's insertion at
    the start boundary and a prefix at the end boundary.
    """
    if not target:
        return False
    n = len(target)
    if len(model_insert) < n:
        return False
    return model_insert[-n:] == target if at_start else model_insert[:n] == target


def cer(ref: str, hyp: str) -> float:
    """Character error rate, whitespace-insensitive.

    Stripping whitespace keeps spacing- and accent-only differences near zero,
    which is what separates a reference error from a normalization artifact.
    Approximated from ``SequenceMatcher`` coverage rather than a true edit
    distance; callers only threshold it.
    """
    r = "".join(ref.split())
    h = "".join(hyp.split())
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    sm = SequenceMatcher(None, r, h, autojunk=False)
    matched = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in sm.get_opcodes() if tag == "equal")
    return 1.0 - matched / len(r)


# ---------------------------------------------------------------------------
# Edit finding and scoring
# ---------------------------------------------------------------------------


@dataclass
class RefEdit:
    """One span on which the official and corrected references disagree.

    ``kind`` is ``"delete"`` when the official reference carries tokens the
    correction removes or replaces, and ``"insert"`` when the correction adds
    tokens the official reference omits. ``verdict`` maps model name to
    ``"ref"`` (output matches the official reference), ``"consensus"`` (output
    matches the correction) or ``None`` (model not competent on the clip, so not
    charged for this edit).
    """

    kind: str
    position: str
    ref_tokens: list[str]
    ref_indices: list[int]
    corrected_text: str = ""
    verdict: dict[str, str | None] = field(default_factory=dict)
    consensus_cer: float | None = None
    clip_key: str | None = None

    @property
    def text(self) -> str:
        return " ".join(self.ref_tokens)


def find_ref_errors(
    ref_tokens: list[str],
    corrected: str,
    model_hyps: dict[str, str],
    *,
    min_run_len: int = DEFAULTS["min_run_len"],
    include_middle: bool = DEFAULTS["include_middle"],
    min_ref_match: float = DEFAULTS["min_ref_match"],
    min_consensus_cer: float = DEFAULTS["min_consensus_cer"],
) -> list[RefEdit]:
    """Diff one clip's official reference against its correction, and score models.

    ``ref_tokens`` is the normalized, whitespace-tokenized official reference;
    ``corrected`` the normalized human-corrected transcript; ``model_hyps`` maps
    model name to a normalized hypothesis.
    """
    n_ref = len(ref_tokens)
    if not n_ref:
        return []
    min_matched = int(min_ref_match * n_ref)
    # An unalignably distant correction would read as "the reference is wrong
    # everywhere"; skip the clip rather than mine it.
    if matched_count(ref_tokens, corrected) < min_matched:
        return []

    missed = {m: missed_indices(ref_tokens, h) for m, h in model_hyps.items()}
    inserted = {m: insertions_by_anchor(ref_tokens, h) for m, h in model_hyps.items()}
    matched = {m: matched_count(ref_tokens, h) for m, h in model_hyps.items()}

    def eligible(model: str) -> bool:
        return matched.get(model, 0) >= min_matched

    edits: list[RefEdit] = []

    # -- spans the correction removes or replaces -----------------------------
    corrected_missed = missed_indices(ref_tokens, corrected)
    runs: list[list[int]] = []
    cur: list[int] = []
    for i in range(n_ref):
        if i in corrected_missed:
            cur.append(i)
        else:
            if len(cur) >= min_run_len:
                runs.append(cur)
            cur = []
    if len(cur) >= min_run_len:
        runs.append(cur)
    if not include_middle:
        runs = [r for r in runs if r[0] == 0 or r[-1] == n_ref - 1]

    for run in runs:
        span = set(run)
        ref_chunk = " ".join(ref_tokens[i] for i in run)
        spoken = substitution_at(ref_tokens, corrected, span)
        edit_cer = cer(ref_chunk, spoken)
        if edit_cer < min_consensus_cer:
            continue  # normalization artifact, not a reference error

        verdict: dict[str, str | None] = {}
        for model in model_hyps:
            if not eligible(model):
                verdict[model] = None
            elif missed[model] & span:
                verdict[model] = "consensus"
            else:
                verdict[model] = "ref"
        edits.append(
            RefEdit(
                kind="delete",
                position="start" if run[0] == 0 else ("end" if run[-1] == n_ref - 1 else "middle"),
                ref_tokens=[ref_tokens[i] for i in run],
                ref_indices=run,
                corrected_text=spoken,
                verdict=verdict,
                consensus_cer=round(edit_cer, 3),
            )
        )

    # -- spans the correction adds -------------------------------------------
    # Restricted to the two reference boundaries. Interior insertions cannot be
    # anchored reliably: which side of a matched token an inserted word belongs
    # to is an alignment choice, not a fact about the audio.
    corrected_ins = insertions_by_anchor(ref_tokens, corrected)
    for anchor in (0, n_ref):
        at_start = anchor == 0
        chunk = corrected_ins.get(anchor, [])
        if len(chunk) < min_run_len:
            continue
        verdict = {}
        for model in model_hyps:
            if not eligible(model):
                verdict[model] = None
            elif emitted_insertion(inserted[model].get(anchor, []), chunk, at_start):
                verdict[model] = "consensus"
            else:
                verdict[model] = "ref"
        edits.append(
            RefEdit(
                kind="insert",
                position="start" if at_start else "end",
                ref_tokens=chunk,
                ref_indices=[anchor],
                corrected_text=" ".join(chunk),
                verdict=verdict,
            )
        )

    return edits


def ref_error_agreement(edits: list[RefEdit]) -> dict[str, dict]:
    """Aggregate per-model reference-error agreement rates over a collection of edits.

    Returns ``{model: {"rate", "n_ref", "n_eligible", "lo", "hi"}}``, where
    ``rate = n_ref / n_eligible`` and ``lo``/``hi`` are a 95% Wilson interval.
    Models are only charged for edits they were eligible for, so denominators
    differ slightly between models; the count is written alongside the rate.
    """
    tally: dict[str, list[int]] = {}
    for edit in edits:
        for model, verdict in edit.verdict.items():
            if verdict is None:
                continue
            k, n = tally.setdefault(model, [0, 0])
            tally[model] = [k + (verdict == "ref"), n + 1]

    out = {}
    for model, (k, n) in tally.items():
        p, lo, hi = wilson(k, n)
        out[model] = {"rate": p, "n_ref": k, "n_eligible": n, "lo": lo, "hi": hi}
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for ``k`` successes in ``n``.

    Wilson rather than a normal approximation because several models sit near 0,
    where the normal interval leaves the unit range.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, center - half), min(1.0, center + half)
