#!/usr/bin/env python3
# Adapted from the reference implementation accompanying "Quantifying Benchmark
# Optimization in ASR Models" (https://github.com/tlebryk/asr-benchmark-optimization,
# Apache-2.0).
"""Reference rendering agreement: extraction, classification and scoring.

A **flagged span** is a span of a raw reference transcript that the leaderboard's
text normalizer rewrites, so the reference's rendering and the normalizer's
rendering of the same words score identically under WER.

At each flagged span a hypothesis is **eligible** when its normalized text
reproduces the flagged span's normalized words (the model got the words right, so
only the rendering is in question), and it **agrees** when its raw text there is
the same reference rendering after case-folding and removing edge punctuation.
For number words, optional hyphenation is ignored. The agreement rate is the
share of eligible flagged spans at which a model agrees.

A single rate is reported over spelling, pointed acronyms, and numbers. Casing
and punctuation are runner-wide style choices. Normalizer abbreviation rewrites
are excluded because they contain ambiguous homographs and transcript fragments.

Public entry points: :func:`extract_flagged_spans` and :func:`keep_flagged_span`
build a clip's list, :func:`score_clip` scores one hypothesis against it, and
:func:`score_pairs` aggregates over a whole manifest.
"""

from __future__ import annotations

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
    """This repo's English normalizer, as instantiated by ``normalizer.data_utils``.

    The same normalizer the leaderboard's English WER is computed under, so a span
    it rewrites is by construction free under WER. It is not interchangeable with
    the upstream Whisper normalizer, which lacks the acronym, name and compound
    stages: ``U.S.`` normalizes to ``us`` here and to ``u s`` there.
    """
    from normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


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

_BARE_WORD_PATTERN = re.compile(r"^\\b(\w+)\\b$")


@lru_cache(maxsize=1)
def abbrevs() -> frozenset[str]:
    """Titles the normalizer expands to a single word, read off its own table.

    Derived rather than hand-listed so the two cannot drift apart. A replacement
    of more than one word (``wanna`` to ``want to``) is a contraction, not a title.
    """
    out = set()
    for pattern, replacement in _normalizer().replacers.items():
        m = _BARE_WORD_PATTERN.match(pattern)
        if m and len(replacement.split()) == 1:
            out.add(m.group(1))
    return frozenset(out)


@lru_cache(maxsize=1)
def disfluencies() -> frozenset[str]:
    """Fillers the normalizer deletes, read off its own ``ignore_patterns``."""
    inner = _normalizer().ignore_patterns.strip("\\b()")
    return frozenset(inner.split("|"))


ACRONYM_RE = re.compile(r"^(?:[A-Za-z]\.\s*)+[A-Za-z]?\.?$")
EDGE_PUNCT = string.punctuation + "‘’“”–—…"
PUNCT_TBL = str.maketrans("", "", string.punctuation + "‘’“”–—…")
_DIGIT_RE = re.compile(r"\d")
_HYPHEN_RE = re.compile(r"[-‐‑‒–—]+")


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

    First match wins, so the order is part of the definition.
    """
    rl = raw_span.lower()
    if "'" in raw_span or "’" in raw_span:
        if len(norm_span.split()) > len(raw_span.split()):
            return "contraction"
    stripped = rl.translate(PUNCT_TBL).strip()
    if stripped in disfluencies() and norm_span == "":
        return "disfluency"
    if rl == norm_span:
        return "case"
    if re.sub(r"\s+", " ", stripped) == norm_span:
        return "punct"
    spelling = spelling_map()
    if spelling.get(rl) == norm_span or spelling.get(stripped) == norm_span:
        return "spelling"
    if re.fullmatch(r"(" + "|".join(sorted(abbrevs())) + r")\.?", rl):
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

# Only the three interpretable, acoustically equivalent rendering choices reach
# scoring. The remaining classes either reflect what was spoken, runner-wide
# formatting, ambiguous normalizer expansions, or heterogeneous rewrites.
EXCLUDED_CLASSES = frozenset(
    {"contraction", "disfluency", "case", "punct", "abbrev", "other"}
)

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


# Words that may open and close a two-word reading, so that "forty seven" is one
# number while "one one" (which the normalizer glues into "11") is two.
TENS_WORDS = frozenset("twenty thirty forty fifty sixty seventy eighty ninety".split())
UNIT_WORDS = frozenset("one two three four five six seven eight nine".split())
UNIT_ORDINALS = frozenset("first second third fourth fifth sixth seventh eighth ninth".split())


def _one_number_reading(raw: str) -> bool:
    """Whether the raw span is exactly one spoken number, in words.

    A single number word, or a tens word followed by a unit ("forty seven",
    "twenty first"). Anything else in the class is the normalizer gluing two
    separately spoken numbers together — "one one" to "11", "seven five" to "75" —
    which is not one number rendered two ways. Punctuation inside or trailing the
    span ("forty,") makes the rendering a joint punctuation-and-number choice, so
    it is excluded too.
    """
    tokens = _HYPHEN_RE.sub(" ", raw).split()
    if not tokens or any(tok.lower() not in NUMBER_WORDS for tok in tokens):
        return False
    if len(tokens) == 1:
        return True
    if len(tokens) == 2:
        first, second = tokens[0].lower(), tokens[1].lower()
        return first in TENS_WORDS and (second in UNIT_WORDS or second in UNIT_ORDINALS)
    return False

# Spelling spans blocked by their RAW (reference-side) form. These fail the
# audibility test rather than the sense test: the raw form is pronounced
# differently from its expansion, so the audio does determine the choice.
# "'kay" (-> okay) is clipped speech, audibly distinct from "okay"; plain
# "ok"/"OK" -> okay is kept, as both are read identically.
BLOCKED_RAW_SPELLINGS = frozenset({"kay"})

# Classes whose rendering the audio does not determine and which are not a
# transcript-wide house-style convention either. The reported rate is over these.
SCORED_CLASSES = frozenset({"spelling", "acronym", "number"})

# Reported per class, in this order. Case/punctuation measure runner settings;
# abbreviations contain normalizer homographs such as "gen" -> "general".
REPORTED_CLASSES = ("spelling", "acronym", "number")


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
        return _in_number_band(norm) and _one_number_reading(raw)
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


def comparable(span: str, cls: str | None = None) -> str:
    """The form two raw spans are compared in.

    Case and edge punctuation are transcript-wide conventions, not the per-token
    choice being measured, so ``COLOUR``, ``colour`` and ``colour,`` count as the
    same rendering of ``colour``. Without this an all-uppercase or unpunctuated
    model scores zero on every span for a reason the rate does not claim to
    measure.
    """
    out = span.casefold().strip(EDGE_PUNCT)
    if cls == "number":
        # The measured choice is words versus digits, not optional hyphenation:
        # "fifty-two" and "fifty two" are the same words-side rendering.
        out = _HYPHEN_RE.sub(" ", out)
        out = re.sub(r"\s+", " ", out).strip()
    return out


def score_clip(flagged_spans, ref_full, hyp: str) -> list[tuple[int, bool, bool, str]]:
    """Score one hypothesis at every flagged span of one clip.

    Returns ``(span_index, eligible, agreed, hyp_raw_span)`` per span. A span is
    eligible when its normalized reference span falls inside a block the
    reference-to-hypothesis normalized alignment marks equal, and the aligned raw
    hypothesis text normalizes to exactly that span — the model produced those
    words and nothing extra. Agreement then compares the two raw spans under
    :func:`comparable`.
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
            # No raw hypothesis token maps to those normalized words, so there is
            # nothing to compare; not eligible rather than an automatic miss.
            res.append((si, False, False, ""))
            continue
        a, b = min(s[0] for s in spans), max(s[1] for s in spans)
        hyp_raw = " ".join(hraw[a : b + 1])
        # The aligned raw hypothesis may cover more than the span's words, as when
        # "twenty" aligns inside "G20" or "eighteen" inside "18-year-old". Those
        # two renderings are not comparable, so the span is not eligible.
        if normalize(hyp_raw) != norm_span:
            res.append((si, False, False, hyp_raw))
            continue
        res.append((si, True, comparable(hyp_raw, cls) == comparable(raw_span, cls), hyp_raw))
    return res


def score_pairs(pairs) -> dict:
    """Aggregate rendering agreement over ``(reference, hypothesis)`` pairs.

    Returns ``scored`` (the reported pool) as ``[agreed, eligible]`` plus the same
    pair per class, for inspection. Clips with no retained flagged span contribute
    nothing.
    """
    scored = [0, 0]
    by_class = {cls: [0, 0] for cls in REPORTED_CLASSES}
    for ref, hyp in pairs:
        # One malformed row must not abort a dataset: 82 contributed manifests are
        # not uniformly well-formed.
        if not isinstance(ref, str) or not isinstance(hyp, str):
            continue
        flagged_spans, ref_full = flagged_spans_for(" ".join(ref.split()))
        if not flagged_spans:
            continue
        per_span = score_clip(list(flagged_spans), list(ref_full), hyp)
        for (_, eligible, agreed, _), fspan in zip(per_span, flagged_spans):
            if not eligible:
                continue
            cls = fspan[2]
            if cls in by_class:
                by_class[cls][1] += 1
                by_class[cls][0] += agreed
            if cls in SCORED_CLASSES:
                scored[1] += 1
                scored[0] += agreed
    return {"scored": scored, "by_class": by_class}
