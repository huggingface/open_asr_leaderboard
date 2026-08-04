#!/usr/bin/env python3
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
# Apache-2.0).
"""Reference rendering agreement: extraction, classification and scoring.

A **flagged span** is a span of a raw reference transcript that the leaderboard's
text normalizer rewrites, so the reference's rendering and the normalizer's
rendering of the same words score identically under WER. Casing, punctuation,
en-GB/en-US spelling, honorific abbreviations, pointed acronyms and
digits-versus-words are all such rewrites.

At each flagged span a hypothesis is **eligible** when its normalized text
reproduces the flagged span's normalized words (the model got the words right, so
only the rendering is in question), and it **agrees** when its raw text there is
character-for-character the reference's raw span. The agreement rate is the share
of eligible flagged spans at which a model agrees.

A single rate is reported, taken over :data:`SCORED_CLASSES` — the classes where
the choice is recoverable from neither the audio nor a transcript-wide convention.
Casing and punctuation are labelled and counted but excluded from it, since they
are house style rather than per-token choices.

Public entry points: :func:`extract_flagged_spans` and :func:`keep_flagged_span`
build a clip's list, :func:`score_clip` scores one hypothesis against it, and
:func:`score_pairs` aggregates over a whole manifest.
"""

from __future__ import annotations

import math
import os
import re
import string
import sys
from difflib import SequenceMatcher
from functools import lru_cache

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _make_normalizer():
    """The Whisper English normalizer, with this repo's en-GB/en-US spelling map.

    The same normalizer the leaderboard's English WER is computed under, so a
    span it rewrites is by construction free under WER.
    """
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    from normalizer.english_abbreviations import english_spelling_normalizer

    return EnglishTextNormalizer(english_spelling_mapping=english_spelling_normalizer)


@lru_cache(maxsize=1)
def _normalizer():
    return _make_normalizer()


@lru_cache(maxsize=1)
def spelling_map() -> dict[str, str]:
    """The en-GB to en-US map the normalizer applies, used to label flagged spans."""
    from normalizer.english_abbreviations import english_spelling_normalizer

    return english_spelling_normalizer


# Parenthesized spans are annotations rather than speech; the Whisper normalizer
# strips the halfwidth form only, so both forms are removed up front.
_PAREN_RE = re.compile(r"[(（][^)）]+?[)）]")

# Whitespace does not delimit words in CJK, so a CJK run — which a multilingual
# model occasionally emits on an English benchmark — is spaced out per character,
# making the alignment over it character-wise rather than one opaque token.
_CJK_RE = re.compile(r"([㐀-䶿一-鿿぀-ゟ゠-ヿ가-힯])")


def normalize(text: str) -> str:
    if not text:
        return ""
    out = _normalizer()(_PAREN_RE.sub("", text)).strip()
    return re.sub(r"\s+", " ", _CJK_RE.sub(r" \1 ", out)).strip()


@lru_cache(maxsize=1 << 20)
def ntok(token: str) -> tuple[str, ...]:
    """Normalized words produced by a single raw whitespace token on its own."""
    return tuple(normalize(token).split())


def nfull(text: str) -> list[str]:
    """Normalized words produced by the whole string at once."""
    return normalize(text).split()


# ---------------------------------------------------------------------------
# Flagged-span extraction
# ---------------------------------------------------------------------------

# Honorifics and units the normalizer's abbreviation table expands.
ABBREVS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "jr",
    "sr",
    "prof",
    "capt",
    "gov",
    "gen",
    "sen",
    "rep",
    "rev",
    "hon",
    "esq",
    "ltd",
    "col",
    "ft",
}
# Filler tokens the normalizer deletes outright.
DISFL = {"uh", "um", "hmm", "mm", "mhm", "mmm", "huh", "ah", "er"}
ACRONYM_RE = re.compile(r"^(?:[A-Za-z]\.\s*)+[A-Za-z]?\.?$")
PUNCT_TBL = str.maketrans("", "", string.punctuation + "‘’“”–—…")
_DIGIT_RE = re.compile(r"\d")


def group_tokens(text: str):
    """Map the raw whitespace tokens of ``text`` onto its normalized words.

    Returns ``(raw_toks, full, groups)`` where ``groups`` is a list of
    ``(raw_lo, raw_hi_inclusive, full_lo, full_hi_exclusive)``. Contiguous raw
    tokens whose per-token normalization disagrees with the full-string
    normalization are merged into a single group, since a normalizer rule that
    spans a token boundary (``"1 000"`` to ``"1000"``) has no per-token
    alignment. ``full_lo = full_hi = -1`` marks a span the normalizer deleted.
    """
    raw = text.split()
    full = nfull(text)
    if not raw:
        return raw, full, []
    pertok = [ntok(t) for t in raw]
    words, owner = [], []
    for i, ws in enumerate(pertok):
        for w in ws:
            words.append(w)
            owner.append(i)

    # raw index -> normalized indices (from equal blocks); diffs become merge regions
    r2f: dict[int, list[int]] = {i: [] for i in range(len(raw))}
    regions: list[tuple[list[int], int, int]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, words, full, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                r2f[owner[i1 + k]].append(j1 + k)
        else:
            regions.append((sorted(set(owner[i1:i2])), j1, j2))

    # Union-find over raw indices; a diff region welds its whole raw span together.
    parent = list(range(len(raw)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    reg_full: dict[int, list[int]] = {}
    orphan_full: list[tuple[int, int]] = []
    for rset, j1, j2 in regions:
        if not rset:
            orphan_full.append((j1, j2))
            continue
        lo, hi = rset[0], rset[-1]
        for i in range(lo, hi):
            union(i, i + 1)
        reg_full.setdefault(lo, []).extend([j1, j2 - 1] if j2 > j1 else [])

    members: dict[int, list[int]] = {}
    for i in range(len(raw)):
        members.setdefault(find(i), []).append(i)
    groups = []
    for root, idxs in sorted(members.items()):
        lo, hi = min(idxs), max(idxs)
        fs = []
        for i in idxs:
            fs.extend(r2f[i])
        for k in idxs:
            fs.extend(reg_full.get(k, []))
        if fs:
            groups.append((lo, hi, min(fs), max(fs) + 1))
        else:
            groups.append((lo, hi, -1, -1))  # normalizer deleted this span

    # Normalized words with no raw source (insertions) attach to the preceding group.
    for j1, j2 in orphan_full:
        best = None
        for gi, (lo, hi, fa, fb) in enumerate(groups):
            if fb != -1 and fb <= j1:
                best = gi
        if best is not None:
            lo, hi, fa, fb = groups[best]
            groups[best] = (lo, hi, fa, max(fb, j2))
    return raw, full, groups


def classify(raw_span: str, norm_span: str) -> str:
    """Label the rewrite that turns ``raw_span`` into ``norm_span``.

    First match wins, so the order is part of the definition; see the ordering
    note in ``README.md``.
    """
    rl = raw_span.lower()
    if "'" in raw_span or "’" in raw_span:
        if len(norm_span.split()) > len(raw_span.split()):
            return "contraction"
    stripped = rl.translate(PUNCT_TBL).strip()
    if stripped in DISFL and norm_span == "":
        return "disfluency"
    if rl == norm_span:
        return "case"
    if re.sub(r"\s+", " ", stripped) == norm_span:
        return "punct"
    spelling = spelling_map()
    if spelling.get(rl) == norm_span or spelling.get(stripped) == norm_span:
        return "spelling"
    if re.fullmatch(r"(" + "|".join(sorted(ABBREVS)) + r")\.?", rl):
        return "abbrev"
    if ACRONYM_RE.match(raw_span):
        return "acronym"
    rd, nd = bool(_DIGIT_RE.search(raw_span)), bool(_DIGIT_RE.search(norm_span))
    if rd != nd:
        return "number"
    return "other"


def extract_flagged_spans(ref: str) -> list[tuple[str, str, str, int, int, int, int]]:
    """Every span of ``ref`` the normalizer rewrites.

    Each flagged span is ``(raw_span, norm_span, cls, raw_lo, raw_hi, full_lo,
    full_hi)``; the raw bounds are inclusive, the normalized bounds are a
    half-open slice of ``nfull(ref)``.
    """
    raw, full, groups = group_tokens(ref)
    out = []
    for lo, hi, fa, fb in groups:
        raw_span = " ".join(raw[lo : hi + 1])
        norm_span = " ".join(full[fa:fb]) if fa >= 0 else ""
        if raw_span == norm_span:
            continue
        out.append((raw_span, norm_span, classify(raw_span, norm_span), lo, hi, fa, fb))
    return out


# ---------------------------------------------------------------------------
# Frozen filters
# ---------------------------------------------------------------------------

# The audio determines these, so reproducing the reference's rendering is not a
# formatting choice: whether a contraction was spoken contracted, and whether a
# filler was uttered at all.
EXCLUDED_CLASSES = frozenset({"contraction", "disfluency"})

# en-GB/en-US pairs whose American form is also an ordinary English word with a
# different sense, so the reference's spelling is forced by meaning rather than
# chosen. Keyed by the normalized (American) form.
BLOCKED_SPELLINGS = frozenset(
    {
        "check",  # cheque: every observed use is the verb or "checks and balances"
        "checks",
        "program",  # programme: the software sense is spelled "program" in en-GB too
        "programs",
        "connection",  # connexion: "connexion" is archaic, not a live variant
        "connections",
        "ton",  # tonne: "tons of" is an idiom, not a unit
        "tons",
        "practice",  # practise: en-GB spells the noun "practice"; only the verb differs
        "practices",
        "draft",  # draught: a draft document is a different sense entirely
        "drafts",
        "story",  # storey: the narrative sense is spelled "story" in en-GB too
        "stories",
        "meter",  # metre: the measuring-device sense is spelled "meter" in en-GB too
        "meters",
        "filter",  # philtre: a philtre is a love potion, an unrelated word
        "filters",
        "biased",  # biassed: "biassed" is vanishingly rare in either dialect
        "curb",  # kerb: not in this repo's map, blocked defensively
        "curbs",
        "tire",  # tyre: the verb "to tire" is spelled the same in en-GB
        "tires",
    }
)

# A number span is kept only in the magnitude band with exactly one natural
# spoken form, so the reference's choice was how to write it rather than what was
# said. Below eleven, spelling the number out is the near-uniform convention of
# edited prose, so the reference is following a rule; at four digits and above,
# "1984" is read either "nineteen eighty-four" or "one thousand nine hundred
# eighty-four", and three-digit forms split the same way ("999" as "nine
# ninety-nine" or "nine hundred ninety-nine").
NUMBER_BAND = range(11, 100)

# Words a kept number span may be built from. Anything else in the raw span means
# the normalizer merged across a boundary rather than rewriting one number:
# "six o'clock" to "60 clock", "one third" to "13rd", "eleven one" to "111".
NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
    sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety
    first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth
    fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth
    fiftieth sixtieth seventieth eightieth ninetieth""".split()
)

_BARE_INT_RE = re.compile(r"^\d+$")
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")


def _ordinal_suffix(value: int) -> str:
    if value % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _in_number_band(norm: str) -> bool:
    """Whether the normalized form is a bare integer or ordinal inside the band.

    An ordinal whose suffix disagrees with its digits (``13rd``) is a merge of two
    separate numbers rather than one ordinal, so it is rejected.
    """
    n = norm.strip()
    if _BARE_INT_RE.match(n):
        return int(n) in NUMBER_BAND
    m = _ORDINAL_RE.match(n)
    if not m:
        return False
    value = int(m.group(1))
    return value in NUMBER_BAND and m.group(2) == _ordinal_suffix(value)


def _all_number_words(raw: str) -> bool:
    """Whether the raw span is number words only, with no punctuation or digits.

    Punctuation inside or trailing the span (``"forty,"``) makes the rendering a
    joint punctuation-and-number choice, which the number class should not carry.
    """
    for token in raw.replace("-", " ").split():
        if token.lower() not in NUMBER_WORDS:
            return False
    return bool(raw.strip())

# Spelling spans blocked by their RAW (reference-side) form. These fail the
# audibility test rather than the sense test: the raw form is pronounced
# differently from its expansion, so the audio does determine the choice.
# "'kay" (-> okay) is clipped speech, audibly distinct from "okay"; plain
# "ok"/"OK" -> okay is kept, as both are read identically.
BLOCKED_RAW_SPELLINGS = frozenset({"kay"})

# Classes whose rendering the audio does not determine and which are not a
# transcript-wide house-style convention either. The reported rate is over these.
SCORED_CLASSES = frozenset({"spelling", "abbrev", "acronym", "number"})

# Reported per class, in this order.
REPORTED_CLASSES = ("case", "punct", "spelling", "abbrev", "acronym", "number", "other")


def keep_flagged_span(fspan) -> bool:
    """Whether a flagged span survives the frozen exclusions."""
    raw, norm, cls = fspan[0], fspan[1], fspan[2]
    if cls in EXCLUDED_CLASSES:
        return False
    if cls == "spelling":
        if norm.lower() in BLOCKED_SPELLINGS:
            return False
        if raw.lower().strip(".,!?;:'\"\u2019") in BLOCKED_RAW_SPELLINGS:
            return False
    if cls == "number":
        return _in_number_band(norm) and _all_number_words(raw)
    return True


@lru_cache(maxsize=200_000)
def flagged_spans_for(ref: str) -> tuple[tuple, tuple[str, ...]]:
    """Retained flagged spans of ``ref`` plus its normalized words.

    ``ref`` must have its whitespace collapsed already, so that manifests that
    differ only in whitespace share a cache entry.
    """
    flagged_spans = tuple(s for s in extract_flagged_spans(ref) if keep_flagged_span(s))
    return flagged_spans, tuple(nfull(ref))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_clip(flagged_spans, ref_full, hyp: str) -> list[tuple[int, bool, bool, str]]:
    """Score one hypothesis at every flagged span of one clip.

    Returns ``(span_index, eligible, agreed, hyp_raw_span)`` per span. A span is
    eligible when its normalized reference span falls inside a block the
    reference-to-hypothesis normalized alignment marks equal, i.e. the model
    produced those words; the raw hypothesis span aligned to it is then compared
    to the reference's raw span for exact string equality.
    """
    hraw, hfull, hgroups = group_tokens(hyp)
    # normalized hypothesis word index -> raw hypothesis token span
    hmap: dict[int, tuple[int, int]] = {}
    for lo, hi, fa, fb in hgroups:
        if fa < 0:
            continue
        for j in range(fa, fb):
            if j in hmap:
                a, b = hmap[j]
                hmap[j] = (min(a, lo), max(b, hi))
            else:
                hmap[j] = (lo, hi)

    equals = [
        (i1, i2, j1)
        for tag, i1, i2, j1, _ in SequenceMatcher(None, list(ref_full), hfull, autojunk=False).get_opcodes()
        if tag == "equal"
    ]
    res = []
    for si, (raw_span, norm_span, cls, lo, hi, fa, fb) in enumerate(flagged_spans):
        if fa < 0 or fb <= fa:
            res.append((si, False, False, ""))
            continue
        blk = next((e for e in equals if e[0] <= fa and fb <= e[1]), None)
        if blk is None:
            res.append((si, False, False, ""))
            continue
        i1, _, j1 = blk
        hj1, hj2 = j1 + (fa - i1), j1 + (fb - i1)
        spans = [hmap[j] for j in range(hj1, hj2) if j in hmap]
        if not spans:
            res.append((si, True, False, ""))
            continue
        a, b = min(s[0] for s in spans), max(s[1] for s in spans)
        hyp_raw = " ".join(hraw[a : b + 1])
        res.append((si, True, hyp_raw == raw_span, hyp_raw))
    return res


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval on ``k / n``."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def score_pairs(pairs) -> dict:
    """Aggregate rendering agreement over ``(reference, hypothesis)`` pairs.

    Returns ``scored`` (the reported pool) as ``[agreed, eligible]``, ``all``
    (every retained class) and the same pair per class, both for inspection only.
    Clips with no retained flagged span contribute nothing.
    """
    every = [0, 0]
    scored = [0, 0]
    by_class = {cls: [0, 0] for cls in REPORTED_CLASSES}
    for ref, hyp in pairs:
        flagged_spans, ref_full = flagged_spans_for(" ".join(ref.split()))
        if not flagged_spans:
            continue
        per_span = score_clip(list(flagged_spans), list(ref_full), hyp)
        for (_, eligible, agreed, _), fspan in zip(per_span, flagged_spans):
            if not eligible:
                continue
            every[1] += 1
            every[0] += agreed
            cls = fspan[2]
            if cls in by_class:
                by_class[cls][1] += 1
                by_class[cls][0] += agreed
            if cls in SCORED_CLASSES:
                scored[1] += 1
                scored[0] += agreed
    return {"scored": scored, "all": every, "by_class": by_class}
