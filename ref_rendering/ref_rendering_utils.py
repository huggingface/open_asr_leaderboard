"""Curated reference-rendering extraction and agreement scoring."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from normalizer import EnglishTextNormalizer
from normalizer.english_abbreviations import english_spelling_normalizer

NORMALIZER = EnglishTextNormalizer()
REPORTED_CLASSES = ("spelling", "initialism", "number", "lexical")


def normalize(text: str) -> str:
    return " ".join(NORMALIZER(text).split()) if text else ""


@dataclass(frozen=True)
class Candidate:
    raw: str
    norm: str
    cls: str
    group: str
    arm: str
    raw_lo: int
    raw_hi: int
    full_lo: int
    full_hi: int


@dataclass(frozen=True)
class RawMatch:
    start: int
    end: int
    cls: str
    group: str
    arm: str


def _nonoverlap(matches):
    out = []
    for match in sorted(
        matches, key=lambda item: (item.start, -(item.end - item.start))
    ):
        if out and match.start < out[-1].end:
            continue
        out.append(match)
    return out


# The normalizer supplies the American counterpart; this whitelist decides which
# acoustically equivalent families are probes. It deliberately excludes mappings
# whose pronunciation or meaning can change (for example aluminium/aluminum,
# cheque/check, programme/program, and tyre/tire).
SPELLING_PREFIXES = tuple("behaviour centre colour defence favour honour minimi mould neighbour organi realis recogni theatre travell".split())
SPELLING_KEYS = frozenset(
    """accoutrements aesthetic aesthetically aesthetics analogue apologise ardour armour armoured authorised axe bannister
    cancelled cancellation cancellations catalogues chilli civilised councillor counselled criticise criticised demeanour
    democratisation destabilise dialogue dialogues emphasise emphasised encyclopaedia endeavour endeavoured fibre finalised
    fertiliser fulfil fulfilment globalisation globalised gravelled grey harbour harmonisation humour humours ionising labelled
    labelling labour labourers labours levelled litre lustre manoeuvring marshalled marvelled marvellous mitre monologue offence
    offences omelette orthopaedic pencilled pretence revelled rumour saviour scepticism sepulchre shrivelled signalled sombre
    specialisation specialised spectre stabilisation sulphur summarise towelling tranquillity vapours""".split()
)
SPELLING_PAIRS = {
    british: american
    for british, american in english_spelling_normalizer.items()
    if british.startswith(SPELLING_PREFIXES) or british in SPELLING_KEYS
}


SPELLING_FORMS = {}
for _british, _american in SPELLING_PAIRS.items():
    _group = "spell:" + _american
    SPELLING_FORMS[_british] = (_group, "british")
    SPELLING_FORMS.setdefault(_american, (_group, "american"))
SPELLING_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(map(re.escape, sorted(SPELLING_FORMS, key=len, reverse=True)))
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)


INITIALISMS = frozenset("api bdb dfs dil dnlg dvd hci lc lcd msc nxt pcb pdf rsi sms spnlp tfidf tnt tv uid uv vcr xml xsl xslt".split())


INITIALISM_PATTERNS = (
    re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.\s*)+[A-Za-z]\.?(?![A-Za-z])"),
    re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\s+)+[A-Za-z](?![A-Za-z])"),
    re.compile(
        r"(?<![A-Za-z])(?:"
        + "|".join(sorted(INITIALISMS, key=len, reverse=True))
        + r")(?![A-Za-z])",
        re.IGNORECASE,
    ),
)


LEXICAL = {
    "mister": {"expanded": r"mister", "abbrev": r"mr\.?"},
    "email": {"split": r"e(?:\s+|[-‐‑‒–—])mail", "joined": r"email"},
    "ecommerce": {"split": r"e(?:\s+|[-‐‑‒–—])commerce", "joined": r"ecommerce"},
    "etcetera": {"split": r"et\s+cetera", "joined": r"etcetera", "abbrev": r"etc\.?"},
    "wifi": {"split": r"wi(?:\s+|[-‐‑‒–—])fi", "joined": r"wifi"},
    "scifi": {"split": r"sci(?:\s+|[-‐‑‒–—])fi", "joined": r"scifi"},
    "xray": {"split": r"x(?:\s+|[-‐‑‒–—])ray", "joined": r"xray"},
    "tshirt": {"split": r"t(?:\s+|[-‐‑‒–—])shirt", "joined": r"tshirt"},
}


def _lexical_patterns():
    return [
        ("lexical", "lexical:" + family, arm, body)
        for family, arms in LEXICAL.items()
        for arm, body in arms.items()
    ]


PATTERNS = tuple(
    (
        cls,
        group,
        arm,
        re.compile(r"(?<![A-Za-z])(?:" + body + r")(?![A-Za-z])", re.IGNORECASE),
    )
    for cls, group, arm, body in _lexical_patterns()
)


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = dict(zip(range(20, 100, 10), "twenty thirty forty fifty sixty seventy eighty ninety".split()))
ORD_ONES = "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split()
ORD_TENS = dict(zip(range(20, 100, 10), "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth".split()))


def _cardinal(value):
    if value < len(ONES):
        return ONES[value]
    tens, unit = divmod(value, 10)
    return TENS[tens * 10] if not unit else TENS[tens * 10] + " " + ONES[unit]


def _ordinal(value):
    if value < len(ORD_ONES):
        return ORD_ONES[value]
    if value in ORD_TENS:
        return ORD_TENS[value]
    tens, unit = divmod(value, 10)
    return TENS[tens * 10] + " " + ORD_ONES[unit]


def _suffix(value):
    if value % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


WORD_NUMBER_FORMS = {}
for value in range(11, 100):
    for kind, form in (("cardinal", _cardinal(value)), ("ordinal", _ordinal(value))):
        WORD_NUMBER_FORMS[form] = (value, kind)
WORD_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(
        re.escape(form).replace(r"\ ", r"(?:\s+|[-‐‑‒–—])")
        for form in sorted(WORD_NUMBER_FORMS, key=len, reverse=True)
    )
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)
DIGIT_NUMBER_RE = re.compile(
    r"(?<![\w$€£¥.\-])(\d{1,2})(st|nd|rd|th)?(?![\w%$€£¥.\-])", re.IGNORECASE
)


def _raw_matches(text: str):
    matches = []
    for match in SPELLING_RE.finditer(text):
        group, arm = SPELLING_FORMS[match.group().casefold()]
        matches.append(RawMatch(match.start(), match.end(), "spelling", group, arm))
    matches += [
        RawMatch(match.start(), match.end(), cls, group, arm)
        for cls, group, arm, pattern in PATTERNS
        for match in pattern.finditer(text)
    ]
    for pattern in INITIALISM_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group()
            value = re.sub(r"[.\s]", "", raw).casefold()
            if value not in INITIALISMS:
                continue
            if "." in raw:
                arm = "spaced+dotted" if re.search(r"\.\s+", raw) else "compact+dotted"
            elif re.search(r"\s", raw):
                arm = "spaced+bare"
            else:
                arm = "joined"
            matches.append(
                RawMatch(
                    match.start(), match.end(), "initialism", "initialism:" + value, arm
                )
            )
    for match in WORD_NUMBER_RE.finditer(text):
        form = re.sub(r"(?:\s+|[-‐‑‒–—])", " ", match.group().casefold())
        value, kind = WORD_NUMBER_FORMS[form]
        matches.append(
            RawMatch(
                match.start(), match.end(), "number", f"number:{kind}:{value}", "words"
            )
        )
    for match in DIGIT_NUMBER_RE.finditer(text):
        value, raw_suffix = int(match.group(1)), match.group(2)
        if not 11 <= value <= 99:
            continue
        if raw_suffix and raw_suffix.casefold() != _suffix(value):
            continue
        before = text[max(0, match.start() - 30) : match.start()]
        after = text[match.end() : match.end() + 20]
        if (
            re.search(r"['’]\s*$", before)
            or re.search(
                r"(?:fiscal\s+year|fy|year)\s*['’]?\s*$", before, re.IGNORECASE
            )
            or re.match(r"\s*(?:fiscal\s+year|year)\b", after, re.IGNORECASE)
        ):
            continue
        kind = "ordinal" if raw_suffix else "cardinal"
        matches.append(
            RawMatch(
                match.start(), match.end(), "number", f"number:{kind}:{value}", "digits"
            )
        )
    return _nonoverlap(matches)


def _marker(group: str) -> str:
    """An alphabetic token stable under the English normalizer."""
    chars = []
    for char in group:
        if char.isalpha():
            chars.append(char)
        elif char.isdigit():
            chars.extend(("q", chr(ord("a") + int(char))))
        else:
            chars.append("z")
    return "zzrender" + "".join(chars) + "zz"


def _transform(text: str, *, reference: bool = False):
    """Replace recognized variants with family markers, then normalize."""
    matches = _raw_matches(text)
    if reference:
        # Bare ``mr`` is useful in unpunctuated hypotheses but ambiguous in a
        # reference; only ``Mr.`` and expanded ``mister`` define probes.
        matches = [
            match
            for match in matches
            if not (
                match.group == "lexical:mister"
                and match.arm == "abbrev"
                and "." not in text[match.start : match.end]
            )
        ]
    token_spans = list(re.finditer(r"\S+", text))
    chunks, cursor, candidates = [], 0, []
    for match in matches:
        chunks += [text[cursor : match.start], " ", _marker(match.group), " "]
        touched = [
            i
            for i, token in enumerate(token_spans)
            if token.start() < match.end and match.start < token.end()
        ]
        candidates.append(
            (match, min(touched), max(touched), text[match.start : match.end])
        )
        cursor = match.end
    chunks.append(text[cursor:])
    words = normalize("".join(chunks)).split()

    positioned, candidate_index = [], 0
    for word_index, word in enumerate(words):
        if candidate_index >= len(candidates):
            break
        match, raw_lo, raw_hi, raw_span = candidates[candidate_index]
        if word != _marker(match.group):
            continue
        positioned.append(
            Candidate(
                raw_span,
                normalize(raw_span),
                match.cls,
                match.group,
                match.arm,
                raw_lo,
                raw_hi,
                word_index,
                word_index + 1,
            )
        )
        candidate_index += 1
    if candidate_index != len(candidates):
        raise RuntimeError(f"candidate marker lost during normalization: {text!r}")
    return tuple(positioned), tuple(words)


@lru_cache(maxsize=200_000)
def flagged_spans_for(text: str):
    return _transform(text, reference=True)


def score_clip(candidates, ref_full, hypothesis: str):
    """Return ``(index, eligible, agreed, raw_hypothesis_span)`` per candidate."""
    hcandidates, hfull = _transform(hypothesis)
    hmap = {candidate.full_lo: candidate for candidate in hcandidates}

    equal = [
        (i1, i2, j1)
        for tag, i1, i2, j1, _ in SequenceMatcher(
            None, list(ref_full), hfull, autojunk=False
        ).get_opcodes()
        if tag == "equal"
    ]
    out = []
    for index, candidate in enumerate(candidates):
        block = next(
            (
                block
                for block in equal
                if block[0] <= candidate.full_lo and candidate.full_hi <= block[1]
            ),
            None,
        )
        if block is None:
            out.append((index, False, False, ""))
            continue
        i1, _, j1 = block
        hyp_candidate = hmap.get(j1 + candidate.full_lo - i1)
        if hyp_candidate is None or hyp_candidate.group != candidate.group:
            out.append((index, False, False, ""))
            continue
        out.append((index, True, hyp_candidate.arm == candidate.arm, hyp_candidate.raw))
    return out


def score_pairs(pairs) -> dict:
    """Aggregate direct reference-rendering agreement over ``(ref, hyp)`` pairs."""
    scored = [0, 0]
    by_class = {cls: [0, 0] for cls in REPORTED_CLASSES}
    for reference, hypothesis in pairs:
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            continue
        candidates, ref_full = flagged_spans_for(" ".join(reference.split()))
        for (_, eligible, agreed, _), candidate in zip(
            score_clip(candidates, ref_full, hypothesis), candidates
        ):
            if not eligible:
                continue
            cell = by_class[candidate.cls]
            cell[1] += 1
            cell[0] += agreed
            scored[1] += 1
            scored[0] += agreed
    return {"scored": scored, "by_class": by_class}
