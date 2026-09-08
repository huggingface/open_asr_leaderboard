from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ARMENIAN_PUNCTUATION = "։՝՜՛՞՟՚«»"
ARMENIAN_MODIFIER_MARKS = "՛՜՞՟՚"
DASH_PUNCTUATION = "֊‐‑‒–—―−"
WHITESPACE_RE = re.compile(r"\s+")
HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\r\n]+")
TRAILING_SEGMENT_PUNCTUATION_RE = re.compile(r"[.,]+$")
SPACED_ELLIPSIS_RE = re.compile(r"\.[^\S\r\n]+\.[^\S\r\n]+\.")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"[^\S\r\n]+([,.;:`?՝՜՞»։])")
SPACE_AFTER_PUNCTUATION_RE = re.compile(r"([,.;:`՝։])(?!\s|\d|\.|$)")
SPACE_BEFORE_OPENING_QUOTE_RE = re.compile(r"(?<=[^\s«])«")
SPACE_AFTER_OPENING_QUOTE_RE = re.compile(r"«[^\S\r\n]+")
# Include common precomposed Latin letters so preprocessing number
# normalization does not mistake accents (for example, ``é``) for
# punctuation and insert spaces inside otherwise intact words.
WORD_CHARACTER_CLASS = "0-9A-Za-zÀ-ÖØ-öø-ÿԱ-Ֆա-ֆևЁёА-Яа-я"
SPACE_AFTER_CLOSING_QUOTE_RE = re.compile(rf"»(?=[{WORD_CHARACTER_CLASS}])")
LEADING_DASH_BULLET_RE = re.compile(r"^-[^\S\r\n]+")
ARMENIAN_LETTER_CLASS = "Ա-Ֆա-ֆև"
_SENTENCE_CAPITALIZABLE_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿԱ-Ֆա-ֆևЁёА-Яа-я]"
)
_ARMENIAN_NUMBER_MODIFIER_RE = re.compile(
    rf"(?<![{ARMENIAN_LETTER_CLASS}])"
    rf"([{ARMENIAN_LETTER_CLASS}]+(?:[{ARMENIAN_MODIFIER_MARKS}]+[{ARMENIAN_LETTER_CLASS}]+)+)"
    rf"(?![{ARMENIAN_LETTER_CLASS}])"
)
STANDALONE_ARMENIAN_EV_RE = re.compile(
    rf"(?<![{WORD_CHARACTER_CLASS}])Ե[՛՜՞՟՚']*վ(?![{WORD_CHARACTER_CLASS}])",
    re.IGNORECASE,
)
SPACE_BEFORE_ARMENIAN_MODIFIER_RE = re.compile(
    rf"(?<=[{WORD_CHARACTER_CLASS}])\s+([{ARMENIAN_MODIFIER_MARKS}])"
)
SPACE_AFTER_ARMENIAN_MODIFIER_RE = re.compile(
    rf"(?<![{WORD_CHARACTER_CLASS}])([{ARMENIAN_MODIFIER_MARKS}])\s+(?=[{WORD_CHARACTER_CLASS}])"
)
ASCII_TO_ARMENIAN_PUNCTUATION = str.maketrans({":": "։"})
PREPROCESS_TRANSLATION = str.maketrans(
    {
        ":": "։",
        "`": "՝",
        "'": "՛",
        "̃": "՜",
        "֊": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)
DASH_TO_SPACE_TRANSLATION = str.maketrans(
    {character: " " for character in DASH_PUNCTUATION + "-"}
)
QUOTE_REPLACEMENTS = (
    ("‹‹", "«"),
    ("››", "»"),
    ("<<", "«"),
    (">>", "»"),
)

NORMALIZATION_METADATA = {
    "preprocessing": (
        "Unicode NFKC; common scoring punctuation; normalize quote, apostrophe, "
        "ASCII grave accent to Armenian comma, hyphen, spacing, trailing segment "
        "punctuation, capitalization (including a leading dash bullet), and whitespace"
    ),
    "normalization": (
        "Pre-processing plus lowercase, hyphen-to-space, keep Armenian/English/"
        "Russian letters and digits, preserve HH։MM time separators, remove other "
        "punctuation, normalize whitespace"
    ),
    "numbernormalization": (
        "Normalized scoring plus canonicalization of written and spoken Armenian "
        "numbers; preserve technical identifiers and canonicalize clock times as H։MM"
    ),
    "primary": "normalization",
}

__all__ = [
    "ArmenianTextNormalizer",
    "NORMALIZATION_METADATA",
    "normalization",
    "normalization_numbernormalization",
    "numbernormalization",
    "preprocessing",
    "preprocessing_numbernormalization",
]

_DIGIT_TOKEN_RE = re.compile(r"\d+")
_COMMA_GROUPED_INTEGER_RE = re.compile(
    r"(?<![\w.,])([1-9]\d{0,2}(?:,\d{3})+)(?![\w.,])"
)
_DOT_GROUPED_INTEGER_RE = re.compile(
    r"(?<![\w.,․])([1-9]\d{0,2}(?:[.․]\d{3})+)(?![\w.,․])"
)
_SPACE_GROUPED_INTEGER_RE = re.compile(
    r"(?<![\w.,])([1-9]\d{0,2}(?: \d{3})+)(?![\w.,])"
)
_DECIMAL_RE = re.compile(
    r"(?<!\w)(\d+)[.,․](\d+)"
    r"(?P<suffix>-?(?:ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
    r"(?!\d)",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(?<!\w)(2[0-3]|[01]?\d)[:։]([0-5]\d)"
    r"(?:-(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
    r"(?!\d)",
    re.IGNORECASE,
)
_DOTTED_TIME_RE = re.compile(
    r"(?<!\w)(2[0-3]|[01]?\d)[.]([0-5]\d)"
    r"(?:-(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
    r"(?!\d)",
    re.IGNORECASE,
)
_LEADING_ZERO_COMPACT_TIME_RE = re.compile(
    r"(?<!\w)(0\d)([0-5]\d)"
    r"(?:-(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
    r"(?!\d)",
    re.IGNORECASE,
)
_HOUR_PREFIXED_COMPACT_TIME_RE = re.compile(
    r"(ժամը\s+)((?:[01]\d|2[0-3]))([0-5]\d)"
    r"(?:-(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
    r"(?!\d)",
    re.IGNORECASE,
)
_FRACTION_RE = re.compile(
    r"(?<!\w)(\d+)\s*/\s*(\d+)"
    r"(?:-(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
    r"(?!\w)"
)
_DIGIT_ORDINAL_DASH_RE = re.compile(
    r"(?<!\w)(\d+)\s*[-‐‑–—]\s*(?:ին|րդ|երորդ)"
    r"(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն)?(?!\w)",
    re.IGNORECASE,
)
_DIGIT_ORDINAL_ATTACHED_RE = re.compile(
    r"(?<!\w)(\d+)(?:րդ|երորդ)"
    r"(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն)?(?!\w)",
    re.IGNORECASE,
)
_DIGIT_CARDINAL_SUFFIX_RE = re.compile(
    r"(?<!\w)(\d+)-(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն)"
    r"(?!\w)",
    re.IGNORECASE,
)
_DIGIT_CARDINAL_ATTACHED_SUFFIX_RE = re.compile(
    r"(?<!\w)(\d+)(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն)"
    r"(?!\w)",
    re.IGNORECASE,
)
_DIGIT_ADJECTIVAL_SUFFIX_RE = re.compile(
    r"(?<!\w)(\d+)\s*[-‐‑–—]\s*"
    r"(ական(?:ների|ներին|ներից|ներում|ներով|ները|ներ|ից|ով|ում|ին|ի|ը|ն|ս|դ)?)"
    r"(?!\w)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"%\s*(?:[-‐‑–—]\s*"
    r"(ներից|ներում|ների|ներով|ներին|երին|երից|ները|երը|ներ|ից|ում|ով|ին|ի|ը|ն|ս|դ))?",
    re.IGNORECASE,
)
_DOTTED_ALPHANUMERIC_IDENTIFIER_RE = re.compile(
    r"(?<!\w)\d+(?:[.․]\d+)+[A-Za-z]+(?!\w)"
)
_KILOMETERS_PER_HOUR_RE = re.compile(r"(?<![Ա-Ֆա-ֆև])կմ\s*/\s*ժ(?![Ա-Ֆա-ֆև])", re.IGNORECASE)
_MILLION_ABBREVIATION_RE = re.compile(
    r"(?<![Ա-Ֆա-ֆև])մլն(?![Ա-Ֆա-ֆև])", re.IGNORECASE
)
_BILLION_ABBREVIATION_RE = re.compile(
    r"(?<![Ա-Ֆա-ֆև])մլրդ(?![Ա-Ֆա-ֆև])", re.IGNORECASE
)
_SPACED_ADJECTIVAL_NUMBER_RE = re.compile(
    r"(?<!\w)(\d+)\s+"
    r"(ական(?:ների|ներին|ներից|ներում|ներով|ները|ներ|ից|ով|ում|ին|ի|ը|ն|ս|դ)?)"
    r"(?!\w)",
    re.IGNORECASE,
)
_YEAR_SUFFIX_PATTERN = (
    r"ներից|ներում|ներով|ներին|ների|երին|երից|ները|երը|ներ|եր|ից|ում|ով|ին|ի|ը|ն|ս|դ"
)
_YEAR_MARKER_RE = re.compile(
    r"(?<!\w)(?P<number>\d+)\s*(?:"
    rf"թվական(?P<word_suffix>{_YEAR_SUFFIX_PATTERN})?"
    r"|թ\.?"
    rf"(?:(?:\s*[-֊‐‑‒–—―−]\s*|\s+)?(?P<abbreviation_suffix>{_YEAR_SUFFIX_PATTERN}))?"
    r")(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_CANONICAL_YEAR_SUFFIX_RE = re.compile(
    rf"(?<!\w)(?P<number>\d+)թ\.-(?P<suffix>{_YEAR_SUFFIX_PATTERN})(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_CHRISTIAN_ERA_RE = re.compile(
    r"(?<![Ա-Ֆա-ֆև])(?:քրիստոսից\s+հետո|ք\.\s*հ\.?)"
    r"(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_NN_NUMERIC_SUFFIX_HYPHEN_RE = re.compile(
    r"(?<=\d)-(?=(?:(?:երորդ|րդ|ին)(?:ներից|ներում|ների|ներով|ներին|երին|երից|երը|ներ|ից|ում|ով|ի|ը|ն)?|ներով|ներից|ներում|ներին|ների|երին|ական|ում|ով|ի|ին|ը|ն|ից)(?![Ա-Ֆա-ֆև]))",
    re.IGNORECASE,
)
_CANONICAL_PERCENT_RE = re.compile(
    r"(?<!\w)(\d+(?:ամբողջ\d+|բաժին\d+)?)\s+տոկոս"
    r"(ներից|ներում|ների|ներով|ներին|երին|երից|ները|երը|ներ|ից|ում|ով|ին|ի|ը|ն)?"
    r"(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_PERCENT_SIGN_RE = re.compile(
    r"(?<!\w)(?P<number>\d+(?:ամբողջ\d+|բաժին\d+)?)\s*%"
    r"(?:(?:\s*[-‐‑–—]\s*|\s+)(?P<suffix>"
    r"ներից|ներում|ների|ներով|ներին|երին|երից|ները|երը|ներ|ից|ում|ով|ին|ի|ը|ն|ս|դ))?"
    r"(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_CANONICAL_PERCENT_WITH_SUFFIX_RE = re.compile(
    r"(?<!\w)(?P<number>\d+(?:ամբողջ\d+|բաժին\d+)?)%"
    r"(?:-(?P<suffix>"
    r"ներից|ներում|ների|ներով|ներին|երին|երից|ները|երը|ներ|ից|ում|ով|ին|ի|ը|ն|ս|դ))?"
    r"(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_DOLLAR_WORD_RE = re.compile(
    r"(?<![\w$])(?P<number>\d+)\s+դոլ{1,2}ար(?![Ա-Ֆա-ֆև])",
    re.IGNORECASE,
)
_DOLLAR_PREFIX_RE = re.compile(r"(?<![\w$])\$\s*(?P<number>\d+)(?!\w)")
_DOLLAR_SUFFIX_RE = re.compile(r"(?<!\w)(?P<number>\d+)\s*\$(?!\w)")
_CANONICAL_DOLLAR_RE = re.compile(r"(?<!\w)\d+\s+\$(?!\w)")

# Canonical connector strings stay inside a single token so one numeric
# expression contributes one semantic unit to WER.
_DECIMAL_SEPARATOR = "ամբողջ"
_DECIMAL_END = "տասնորդավերջ"
_TIME_SEPARATOR = "անց"
_TIME_PUNCTUATION = "։"
_TIME_PUNCTUATION_PLACEHOLDER = "asrtimeseparator"
_CANONICAL_TIME_SEPARATOR_PLACEHOLDER = "asrcanonicaltime"
_CANONICAL_TIME_SUFFIX_PLACEHOLDER = "asrtimesuffixcanonical"
_CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER = "asrdecimalsuffixcanonical"
_FRACTION_SEPARATOR = "բաժին"
_ORDINAL_MARKER = "հերթականաթիվ"
_UNITS = {
    "զրո": 0,
    "մեկ": 1,
    "երկու": 2,
    "իրեք": 3,
    "երեք": 3,
    "չորս": 4,
    "հինգ": 5,
    "վեց": 6,
    "յոթ": 7,
    "ութ": 8,
    "ինը": 9,
}
_TEENS = {
    "տաս": 10,
    "տասնմեկ": 11,
    "տասներկու": 12,
    "տասներեք": 13,
    "տասնչորս": 14,
    "տասնհինգ": 15,
    "տասնվեց": 16,
    "տասնյոթ": 17,
    "տասնութ": 18,
    "տասնինը": 19,
}
_TENS = {
    "քսան": 20,
    "երեսուն": 30,
    "քառասուն": 40,
    "հիսուն": 50,
    "վաթսուն": 60,
    "յոթանասուն": 70,
    "ութսուն": 80,
    "իննսուն": 90,
}
_SMALL_CARDINALS = {**_UNITS, **_TEENS, **_TENS}
for _tens_word, _tens_value in _TENS.items():
    for _unit_word, _unit_value in _UNITS.items():
        if _unit_value:
            _SMALL_CARDINALS[_tens_word + _unit_word] = (
                _tens_value + _unit_value
            )

_UNITS_BY_VALUE = {value: word for word, value in _UNITS.items()}
_TENS_BY_VALUE = {value: word for word, value in _TENS.items()}
_SMALL_CARDINALS_BY_VALUE = {
    value: word for word, value in _SMALL_CARDINALS.items()
}

_SCALES = {
    "հազար": 1_000,
    "միլիոն": 1_000_000,
    "միլիարդ": 1_000_000_000,
}
_CARDINAL_WORDS = frozenset({*_SMALL_CARDINALS, "հարյուր", *_SCALES})
_NUMBER_SUFFIXES = (
    "ներից",
    "ներում",
    "ների",
    "ներով",
    "երին",
    "երից",
    "երը",
    "ից",
    "ում",
    "ով",
    "ին",
    "ի",
    "ը",
    "ն",
)
_HYPHEN_SUFFIXES = (
    "ներից",
    "ներում",
    "ներով",
    "ներին",
    "ներից",
    "ների",
    "երին",
    "երից",
    "երով",
    "ները",
    "ների",
    "երը",
    "երի",
    "ներ",
    "եր",
    "ից",
    "ով",
    "ում",
    "ին",
    "ի",
    "ը",
    "ն",
    "ս",
    "դ",
)
_HYPHEN_SUFFIX_PATTERN = "|".join(
    sorted(set(_HYPHEN_SUFFIXES), key=len, reverse=True)
)
HYPHENATED_SUFFIX_RE = re.compile(
    rf"(?P<stem>[{WORD_CHARACTER_CLASS}][{ARMENIAN_MODIFIER_MARKS}]*)»?\s*-\s*"
    rf"(?P<suffix>{_HYPHEN_SUFFIX_PATTERN})"
    rf"(?![{WORD_CHARACTER_CLASS}])"
)
PREPROCESSED_HYPHENATED_SUFFIX_RE = re.compile(
    rf"(?P<stem>[{WORD_CHARACTER_CLASS}][{ARMENIAN_MODIFIER_MARKS}]*)"
    rf"(?P<closing_quote>»)?\s*-\s*"
    rf"(?P<suffix>{_HYPHEN_SUFFIX_PATTERN})(?![{WORD_CHARACTER_CLASS}])"
)
_IRREGULAR_INFLECTED_CARDINALS = {
    "մեկը": ("մեկ", "ը"),
    "մեկին": ("մեկ", "ին"),
    "մեկից": ("մեկ", "ից"),
    "երկուսը": ("երկու", "ը"),
    "երկուսին": ("երկու", "ին"),
    "երկուսից": ("երկու", "ից"),
    "տասներկուսն": ("տասներկու", "ն"),
}
_FRACTION_WORDS = {"կես": (1, 2), "կէս": (1, 2), "քառորդ": (1, 4)}
_COMPOUND_HALF_WORDS = {
    "մեկուկես": 1,
    "երկուսուկես": 2,
    "երեքուկես": 3,
    "չորսուկես": 4,
    "հինգուկես": 5,
    "վեցուկես": 6,
    "յոթուկես": 7,
    "ութուկես": 8,
    "իննուկես": 9,
    "տասուկես": 10,
}


@dataclass(frozen=True)
class _NumberPart:
    value: int
    kind: str
    suffix: str | None = None


def _build_ordinal_words() -> dict[str, _NumberPart]:
    words = {
        "առաջին": _NumberPart(1, "small"),
        "երկրորդ": _NumberPart(2, "small"),
        "երրորդ": _NumberPart(3, "small"),
        "չորրորդ": _NumberPart(4, "small"),
        "հինգերորդ": _NumberPart(5, "small"),
        "վեցերորդ": _NumberPart(6, "small"),
        "յոթերորդ": _NumberPart(7, "small"),
        "ութերորդ": _NumberPart(8, "small"),
        "իններորդ": _NumberPart(9, "small"),
        "տասներորդ": _NumberPart(10, "small"),
        "հարյուրերորդ": _NumberPart(100, "hundred"),
    }
    for cardinal, value in _SMALL_CARDINALS.items():
        if value <= 10:
            continue
        if cardinal.endswith("ինը"):
            ordinal = cardinal[: -len("ինը")] + "իններորդ"
        else:
            ordinal = cardinal + "երորդ"
        words[ordinal] = _NumberPart(value, "small")
    for scale, value in _SCALES.items():
        words[scale + "երորդ"] = _NumberPart(value, "scale")
    return words


_ORDINAL_WORDS = _build_ordinal_words()


def preprocessing(
    text: str, *, replace_newlines: bool = False, capitalize_first: bool = True
) -> str:
    """Apply scoring pre-processing before strict WER/CER."""
    processed = unicodedata.normalize("NFKC", str(text))
    processed = SPACED_ELLIPSIS_RE.sub("...", processed)
    for source, target in QUOTE_REPLACEMENTS:
        processed = processed.replace(source, target)
    processed = processed.translate(PREPROCESS_TRANSLATION)
    processed = _SPACE_GROUPED_INTEGER_RE.sub(
        lambda match: match.group(1).replace(" ", ""), processed
    )
    if replace_newlines:
        processed = processed.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    processed = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", processed)
    processed = SPACE_AFTER_PUNCTUATION_RE.sub(r"\1 ", processed)
    processed = SPACE_BEFORE_OPENING_QUOTE_RE.sub(" «", processed)
    processed = SPACE_AFTER_OPENING_QUOTE_RE.sub("«", processed)
    processed = SPACE_AFTER_CLOSING_QUOTE_RE.sub("» ", processed)
    processed = processed.strip()
    processed = TRAILING_SEGMENT_PUNCTUATION_RE.sub("", processed).strip()
    processed = _CHRISTIAN_ERA_RE.sub("Ք.հ.", processed)
    # A leading digit is a complete sentence prefix; do not treat the first
    # later letter as sentence-initial (for example, ``100 մարդ``).
    if capitalize_first and not processed[:1].isdigit():
        bullet = LEADING_DASH_BULLET_RE.match(processed)
        if bullet:
            # A leading dash followed by whitespace is a list-item marker, not
            # the sentence start. Capitalize the first letter of its phrase.
            processed = bullet.group(0) + _capitalize_first_letter(
                processed[bullet.end() :]
            )
        else:
            processed = _capitalize_first_letter(processed)
    return HORIZONTAL_WHITESPACE_RE.sub(" ", processed).strip()


def normalization(text: str) -> str:
    """Apply pre-processing plus normalized WERn/CERn text cleanup."""
    preprocessed = preprocessing(text, replace_newlines=True)
    preprocessed = _TIME_RE.sub(_protect_time_punctuation, preprocessed)
    normalized = _normalize_standalone_armenian_ev(
        _fold_latin_diacritics(preprocessed)
    ).lower()
    normalized = HYPHENATED_SUFFIX_RE.sub(r"\g<stem>\g<suffix>", normalized)
    normalized = normalized.translate(DASH_TO_SPACE_TRANSLATION)
    normalized = _remove_non_word_characters(normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.replace(
        _TIME_PUNCTUATION_PLACEHOLDER, _TIME_PUNCTUATION
    ).strip()


def _normalize_standalone_armenian_ev(text: str) -> str:
    return STANDALONE_ARMENIAN_EV_RE.sub("եւ", text)


def _remove_non_word_characters(text: str) -> str:
    """Keep letters, numbers, dollar signs, combining marks, and whitespace."""
    return "".join(
        character
        for character in text
        if (
            character.isalnum()
            or character == "$"
            or character.isspace()
            or unicodedata.category(character).startswith("M")
        )
    )


def numbernormalization(text: str) -> str:
    """Apply pre-processing, then expand digit notation into Armenian words."""
    return preprocessing_numbernormalization(text)


def preprocessing_numbernormalization(text: str) -> str:
    """Apply pre-processing and canonicalize written and spoken numbers."""
    canonicalized = _CHRISTIAN_ERA_RE.sub(
        "Ք.հ.",
        _canonicalize_preprocessed_numbers(
            preprocessing(text, replace_newlines=True)
        ),
    )
    return _canonicalize_dollars(_canonicalize_percentage_signs(canonicalized))


def normalization_numbernormalization(text: str) -> str:
    """Apply preprocessing, number normalization, then normalized cleanup."""
    number_normalized = preprocessing_numbernormalization(text)

    # Keep canonical clock-time, year-marker, percent, and dollar spelling
    # through the final cleanup.
    # Other NN punctuation is intentionally handled by ordinary normalization.
    # Numeric suffixes are the exception: ``1-ին`` becomes ``1ին`` rather
    # than ``1 ին``. This also applies to numeric grammatical endings such as
    # ``-ի``, ``-ին``, ``-ը``, ``-ն``, ``-րդ`` and inflected ordinal forms
    # such as ``-րդն``/``-ինից``, ``-ից``, ``-ում``, ``-ով``,
    # ``-ների``, ``-ներով``, ``-ներից``, ``-ներում``, ``-երին``, ``-ներին``,
    # ``-երորդ``, and ``-ական``; unrelated hyphens still become spaces below.
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        key = f"asrnumberpunctuation{len(protected)}marker"
        protected[key] = match.group(0)
        return key

    number_normalized = re.sub(
        r"(?<!\w)\d{1,2}։\d{2}"
        r"(?:-(?:ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն))?"
        r"(?!\d)",
        protect,
        number_normalized,
    )

    def protect_percent(match: re.Match[str]) -> str:
        key = f"asrnumberpunctuation{len(protected)}marker"
        protected[key] = f"{match.group('number')}%{match.group('suffix') or ''}"
        return key

    number_normalized = _CANONICAL_PERCENT_WITH_SUFFIX_RE.sub(
        protect_percent, number_normalized
    )

    def protect_year_marker(match: re.Match[str]) -> str:
        key = f"asrnumberpunctuation{len(protected)}marker"
        protected[key] = f"{match.group('number')}թ-{match.group('suffix')}"
        return key

    number_normalized = _CANONICAL_YEAR_SUFFIX_RE.sub(
        protect_year_marker, number_normalized
    )
    number_normalized = _CANONICAL_DOLLAR_RE.sub(protect, number_normalized)
    number_normalized = _NN_NUMERIC_SUFFIX_HYPHEN_RE.sub("", number_normalized)
    number_normalized = re.sub(
        r"(?<=[Ա-Ֆա-ֆև])/(?=[Ա-Ֆա-ֆև])",
        " ",
        number_normalized,
    )
    normalized = normalization(number_normalized)
    for key, value in protected.items():
        normalized = normalized.replace(key, value)
    return normalized


class ArmenianTextNormalizer:
    """Callable Armenian normalizer used by the leaderboard scoring pipeline.

    Written and spoken number forms are canonicalized to the same representation
    so that formatting differences do not count as recognition errors.
    """

    def __call__(self, text: str) -> str:
        return normalization_numbernormalization(text)


def _protect_time_punctuation(match: re.Match[str]) -> str:
    suffix = match.group(3)
    return (
        f"{match.group(1)}{_TIME_PUNCTUATION_PLACEHOLDER}"
        f"{match.group(2)}{suffix or ''}"
    )


def _fold_latin_diacritics(text: str) -> str:
    """Fold decomposable accented Latin letters to their ASCII base letter."""
    output: list[str] = []
    for character in text:
        decomposed = unicodedata.normalize("NFKD", character)
        if decomposed and decomposed[0].isascii() and decomposed[0].isalpha():
            output.append(decomposed[0])
        else:
            output.append(character)
    return "".join(output)


_INTEGER_NOTATION_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[,.]\d{3})+(?![\w.,])|(?<!\w)\d+(?!\w)")


def _expand_digit_notation(text: str) -> str:
    """Expand standalone integer digit notation; leave spoken words unchanged."""
    protected_tokens: list[str] = []

    def protect_token(match: re.Match[str]) -> str:
        protected_tokens.append(match.group(0))
        return chr(0xE000 + len(protected_tokens) - 1)

    text = _DOTTED_ALPHANUMERIC_IDENTIFIER_RE.sub(protect_token, text)
    text = _DOTTED_TIME_RE.sub(_dotted_time_replacement, text)
    text = _TIME_RE.sub(protect_token, text)
    text = _KILOMETERS_PER_HOUR_RE.sub("կիլոմետր ժամ", text)
    text = _PERCENT_RE.sub(_spoken_percent_replacement, text)
    text = _DIGIT_ADJECTIVAL_SUFFIX_RE.sub(
        _spoken_digit_adjectival_replacement, text
    )
    text = _DIGIT_ORDINAL_DASH_RE.sub(_spoken_digit_ordinal_replacement, text)
    text = _DIGIT_ORDINAL_ATTACHED_RE.sub(_spoken_digit_ordinal_replacement, text)
    text = _DIGIT_CARDINAL_SUFFIX_RE.sub(
        _spoken_digit_cardinal_suffix_replacement, text
    )
    text = _DIGIT_CARDINAL_ATTACHED_SUFFIX_RE.sub(
        _spoken_digit_cardinal_suffix_replacement, text
    )

    def replace(match: re.Match[str]) -> str:
        digits = match.group(0).replace(",", "").replace(".", "")
        return _integer_to_armenian_words(int(digits))

    expanded = _INTEGER_NOTATION_RE.sub(replace, text)
    for index, original in enumerate(protected_tokens):
        expanded = expanded.replace(chr(0xE000 + index), original)
    return HORIZONTAL_WHITESPACE_RE.sub(" ", expanded).strip()


def _spoken_percent_replacement(match: re.Match[str]) -> str:
    suffix = (match.group(1) or "").lower()
    return f" տոկոս{suffix}" if suffix else " տոկոս "


def _spoken_digit_cardinal_suffix_replacement(match: re.Match[str]) -> str:
    return _integer_to_armenian_words(int(match.group(1))) + match.group(2).lower()


def _spoken_digit_adjectival_replacement(match: re.Match[str]) -> str:
    return _integer_to_armenian_words(int(match.group(1))) + match.group(2).lower()


def _spoken_digit_ordinal_replacement(match: re.Match[str]) -> str:
    value = int(match.group(1))
    if value != 1 and re.search(r"[-‐‑–—]\s*ին", match.group(0), re.IGNORECASE):
        return _integer_to_armenian_words(value) + "ին" + (match.group(2) or "").lower()
    return _integer_to_armenian_ordinal(value) + (match.group(2) or "").lower()


def _dotted_time_replacement(match: re.Match[str]) -> str:
    return f"{match.group(1)}:{match.group(2)}" + (f"-{match.group(3)}" if match.group(3) else "")


def _integer_to_armenian_ordinal(value: int) -> str:
    if value <= 0:
        return _integer_to_armenian_words(value) + "երորդ"

    irregular = {
        1: "առաջին",
        2: "երկրորդ",
        3: "երրորդ",
        4: "չորրորդ",
        5: "հինգերորդ",
        6: "վեցերորդ",
        7: "յոթերորդ",
        8: "ութերորդ",
        9: "իններորդ",
        10: "տասներորդ",
    }
    if value in irregular:
        return irregular[value]

    cardinal = _integer_to_armenian_words(value)
    head, separator, final = cardinal.rpartition(" ")
    if final.endswith("ինը"):
        final = final[: -len("ինը")] + "իններորդ"
    else:
        final += "երորդ"
    return f"{head}{separator}{final}"


def _integer_to_armenian_words(value: int) -> str:
    if value < 0:
        raise ValueError("Only non-negative integers are supported")
    if value == 0:
        return "զրո"
    if value >= 1_000_000_000_000:
        # Avoid inventing scale vocabulary outside the benchmark's policy.
        return " ".join(_UNITS_BY_VALUE[int(digit)] for digit in str(value))

    scales = (
        (1_000_000_000, "միլիարդ"),
        (1_000_000, "միլիոն"),
        (1_000, "հազար"),
    )
    parts: list[str] = []
    remainder = value
    for scale_value, scale_word in scales:
        count, remainder = divmod(remainder, scale_value)
        if not count:
            continue
        if count != 1 or scale_value >= 1_000_000:
            parts.append(_integer_below_thousand_to_words(count))
        parts.append(scale_word)
    if remainder:
        parts.append(_integer_below_thousand_to_words(remainder))
    return " ".join(parts)


def _integer_below_thousand_to_words(value: int) -> str:
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        if hundreds != 1:
            parts.append(_UNITS_BY_VALUE[hundreds])
        parts.append("հարյուր")
    if remainder in _SMALL_CARDINALS_BY_VALUE:
        if remainder:
            parts.append(_SMALL_CARDINALS_BY_VALUE[remainder])
    elif remainder:
        tens, units = divmod(remainder, 10)
        parts.append(_TENS_BY_VALUE[tens * 10])
        if units:
            parts.append(_UNITS_BY_VALUE[units])
    return " ".join(parts)


def _capitalize_first_letter(text: str) -> str:
    for index, character in enumerate(text):
        if character in ARMENIAN_MODIFIER_MARKS:
            continue
        if _SENTENCE_CAPITALIZABLE_RE.fullmatch(character):
            return text[:index] + character.upper() + text[index + 1 :]
        return text
    return text


def _is_word_token_character(character: str) -> bool:
    return (
        character.isalnum()
        or character in ARMENIAN_MODIFIER_MARKS
        or unicodedata.category(character).startswith("M")
    )


def _tokenize_words_and_punctuation(text: str) -> list[str]:
    """Keep each Unicode word intact; emit punctuation as separate tokens."""
    tokens: list[str] = []
    word: list[str] = []
    for character in text:
        if _is_word_token_character(character):
            word.append(character)
            continue
        if word:
            tokens.append("".join(word))
            word = []
        if not character.isspace():
            tokens.append(character)
    if word:
        tokens.append("".join(word))
    return tokens


def _protect_adjacent_symbols(text: str) -> tuple[str, list[tuple[str, str, bool, bool]]]:
    """Protect symbol adjacency so NN does not create whitespace around it."""
    protected: list[tuple[str, str, bool, bool]] = []
    output: list[str] = []
    for index, character in enumerate(text):
        if not unicodedata.category(character).startswith("S"):
            output.append(character)
            continue
        marker = f"asrsymbol{len(protected)}marker"
        protected.append(
            (
                marker,
                character,
                index > 0 and not text[index - 1].isspace(),
                index + 1 < len(text) and not text[index + 1].isspace(),
            )
        )
        output.append(marker)
    return "".join(output), protected


def _restore_adjacent_symbols(
    text: str, protected: list[tuple[str, str, bool, bool]]
) -> str:
    for marker, character, attach_left, attach_right in protected:
        escaped_marker = re.escape(marker)
        if attach_left:
            text = re.sub(rf"\s*{escaped_marker}", marker, text)
        if attach_right:
            text = re.sub(rf"{escaped_marker}\s*", marker, text)
        text = text.replace(marker, character)
    return text


def _canonicalize_preprocessed_numbers(text: str) -> str:
    identifiers: list[str] = []

    def protect_identifier(match: re.Match[str]) -> str:
        identifiers.append(match.group(0))
        return f"asridentifier{len(identifiers) - 1}marker"

    prepared, protected_symbols = _protect_adjacent_symbols(
        _separate_number_modifiers(text)
    )
    prepared = _DOTTED_ALPHANUMERIC_IDENTIFIER_RE.sub(
        protect_identifier, prepared
    )
    prepared = _MILLION_ABBREVIATION_RE.sub("միլիոն", prepared)
    prepared = _BILLION_ABBREVIATION_RE.sub("միլիարդ", prepared)
    prepared = _protect_digit_notation(prepared)
    tokens = _tokenize_words_and_punctuation(prepared)
    output = _canonicalize_number_tokens(
        [token.lower() for token in tokens], original_tokens=tokens
    )
    joined = " ".join(output)
    joined = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", joined)
    joined = SPACE_AFTER_PUNCTUATION_RE.sub(r"\1 ", joined)
    joined = SPACE_BEFORE_OPENING_QUOTE_RE.sub(" «", joined)
    joined = SPACE_AFTER_OPENING_QUOTE_RE.sub("«", joined)
    joined = SPACE_AFTER_CLOSING_QUOTE_RE.sub("» ", joined)
    joined = SPACE_BEFORE_ARMENIAN_MODIFIER_RE.sub(r"\1", joined)
    joined = SPACE_AFTER_ARMENIAN_MODIFIER_RE.sub(r"\1", joined)
    joined = re.sub(
        rf"(?<=[{WORD_CHARACTER_CLASS}])\s*([-/])\s*(?=[{WORD_CHARACTER_CLASS}])",
        r"\1",
        joined,
    )
    joined = PREPROCESSED_HYPHENATED_SUFFIX_RE.sub(
        lambda match: (
            f"{match.group('stem')}{match.group('closing_quote') or ''}"
            f"-{match.group('suffix')}"
        ),
        joined,
    )
    joined = re.sub(r"([([])\s+", r"\1", joined)
    joined = re.sub(r"\s+([])])", r"\1", joined)
    joined = re.sub(
        rf"{_CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER}\s+(?=[{ARMENIAN_LETTER_CLASS}])",
        _CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER,
        joined,
    )
    joined = _canonicalize_spaced_adjectival_numbers(joined)
    joined = _canonicalize_year_markers(joined, keep_period=True)
    joined = _canonicalize_percentages(joined)
    joined = WHITESPACE_RE.sub(" ", joined).strip()
    joined = joined.replace(
        _CANONICAL_TIME_SEPARATOR_PLACEHOLDER, "։"
    ).replace(_CANONICAL_TIME_SUFFIX_PLACEHOLDER, "-").replace(
        _CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER, "-"
    )
    for index, identifier in enumerate(identifiers):
        joined = joined.replace(f"asridentifier{index}marker", identifier)
    return _restore_adjacent_symbols(joined, protected_symbols)


def _canonicalize_normalized_numbers(text: str) -> str:
    prepared = preprocessing(text, replace_newlines=True).lower()
    # ``normalization`` deliberately removes Armenian punctuation.  Do not
    # preserve emphasis/question marks from number words here: restoring them
    # after numeral conversion would create forms such as ``1՛`` that are not
    # part of the normalized scoring representation.
    prepared = _separate_number_modifiers(prepared)
    identifiers: list[str] = []

    def protect_identifier(match: re.Match[str]) -> str:
        identifiers.append(normalization(match.group(0)))
        return f"asridentifier{len(identifiers) - 1}marker"

    prepared = _DOTTED_ALPHANUMERIC_IDENTIFIER_RE.sub(
        protect_identifier, prepared
    )
    prepared = _MILLION_ABBREVIATION_RE.sub("միլիոն", prepared)
    prepared = _BILLION_ABBREVIATION_RE.sub("միլիարդ", prepared)
    prepared = _protect_digit_notation(prepared)
    normalized = normalization(prepared)
    normalized = _merge_grouped_integer_digits(normalized)
    if normalized:
        normalized = " ".join(_canonicalize_number_tokens(normalized.split()))
    normalized = _canonicalize_spaced_adjectival_numbers(normalized)
    normalized = _canonicalize_year_markers(normalized, keep_period=False)
    normalized = _canonicalize_percentages(normalized)
    normalized = normalized.replace(
        _CANONICAL_TIME_SEPARATOR_PLACEHOLDER, "։"
    ).replace(_CANONICAL_TIME_SUFFIX_PLACEHOLDER, "-").replace(
        _CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER, "-"
    )
    for index, identifier in enumerate(identifiers):
        normalized = normalized.replace(f"asridentifier{index}marker", identifier)
    return normalized


def _separate_number_modifiers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word = match.group(1)
        modifiers = "".join(
            character for character in word if character in ARMENIAN_MODIFIER_MARKS
        )
        bare_word = "".join(
            character for character in word if character not in ARMENIAN_MODIFIER_MARKS
        )
        if (
            _parse_cardinal_word(bare_word.lower()) is None
            and _parse_ordinal_word(bare_word.lower()) is None
        ):
            return word
        # Modifier punctuation attached to a number word is not part of the
        # number: ``հազա՜ր`` must become ``1000``, not ``1000՜``.
        return bare_word

    return _ARMENIAN_NUMBER_MODIFIER_RE.sub(replace, text)


def _canonicalize_spaced_adjectival_numbers(text: str) -> str:
    return _SPACED_ADJECTIVAL_NUMBER_RE.sub(
        lambda match: f"{int(match.group(1))}{match.group(2).lower()}",
        text,
    )


def _canonicalize_year_markers(text: str, *, keep_period: bool) -> str:
    marker = "թ." if keep_period else "թ"

    def replacement(match: re.Match[str]) -> str:
        suffix = match.group("word_suffix") or match.group("abbreviation_suffix")
        return (
            f"{int(match.group('number'))}{marker}"
            + (f"-{suffix.lower()}" if suffix else "")
        )

    return _YEAR_MARKER_RE.sub(replacement, text)


def _canonicalize_percentages(text: str) -> str:
    return _CANONICAL_PERCENT_RE.sub(
        lambda match: f"{match.group(1)}%"
        + (f"-{match.group(2)}" if match.group(2) else ""),
        text,
    )


def _canonicalize_percentage_signs(text: str) -> str:
    """Canonicalize percent-sign spacing and Armenian case suffixes."""
    return _PERCENT_SIGN_RE.sub(
        lambda match: f"{match.group('number')}%"
        + (f"-{match.group('suffix').lower()}" if match.group("suffix") else ""),
        text,
    )


def _canonicalize_dollars(text: str) -> str:
    """Canonicalize Armenian dollar wording and sign placement as ``100 $``."""
    text = _DOLLAR_WORD_RE.sub(lambda match: f"{match.group('number')} $", text)
    text = _DOLLAR_PREFIX_RE.sub(lambda match: f"{match.group('number')} $", text)
    return _DOLLAR_SUFFIX_RE.sub(lambda match: f"{match.group('number')} $", text)


def _canonicalize_number_tokens(
    tokens: list[str], *, original_tokens: list[str] | None = None
) -> list[str]:
    if original_tokens is not None and len(original_tokens) != len(tokens):
        raise ValueError("original_tokens must match tokens")
    output: list[str] = []
    index = 0
    while index < len(tokens):
        spoken_time = _parse_spoken_time(tokens, index)
        if spoken_time is not None:
            replacement, consumed = spoken_time
            output.append(replacement)
            index += consumed
            continue

        spoken_decimal_scale = _parse_spoken_decimal_scale(tokens, index)
        if spoken_decimal_scale is not None:
            replacement, consumed = spoken_decimal_scale
            output.append(replacement)
            index += consumed
            continue

        digit_ordinal = _parse_digit_ordinal(tokens, index)
        if digit_ordinal is not None:
            replacement, consumed = digit_ordinal
            output.extend(replacement)
            index += consumed
            continue

        compound_half = _parse_compound_half(tokens[index])
        if compound_half is not None:
            output.append(compound_half)
            index += 1
            continue

        spoken_fraction = _parse_spoken_fraction(tokens, index)
        if spoken_fraction is not None:
            replacement, consumed = spoken_fraction
            output.extend(replacement)
            index += consumed
            continue

        ordinal = _parse_spoken_ordinal_sequence(tokens, index)
        if ordinal is not None:
            replacement, consumed = ordinal
            output.extend(replacement)
            index += consumed
            continue

        cardinal = _parse_cardinal_sequence(tokens, index)
        if cardinal is not None:
            replacement, consumed = cardinal
            output.extend(replacement)
            index += consumed
            continue

        token = tokens[index]
        if _DIGIT_TOKEN_RE.fullmatch(token):
            output.append(
                token
                if len(token) > 1 and token.startswith("0")
                else str(int(token))
            )
        else:
            output.append(
                original_tokens[index] if original_tokens is not None else token
            )
        index += 1
    return _collapse_numeric_structures(output)


def _protect_digit_notation(text: str) -> str:
    text = _merge_grouped_integer_digits(text)
    text = _LEADING_ZERO_COMPACT_TIME_RE.sub(_compact_time_replacement, text)
    text = _HOUR_PREFIXED_COMPACT_TIME_RE.sub(
        _hour_prefixed_compact_time_replacement, text
    )
    text = _PERCENT_RE.sub(_spoken_percent_replacement, text)
    text = _TIME_RE.sub(_time_replacement, text)
    text = _DOTTED_TIME_RE.sub(_time_replacement, text)
    text = _FRACTION_RE.sub(_fraction_replacement, text)
    text = _DECIMAL_RE.sub(_decimal_replacement, text)
    text = _DIGIT_ADJECTIVAL_SUFFIX_RE.sub(
        _digit_adjectival_replacement, text
    )
    text = _DIGIT_ORDINAL_DASH_RE.sub(_digit_ordinal_replacement, text)
    text = _DIGIT_ORDINAL_ATTACHED_RE.sub(_digit_ordinal_replacement, text)
    text = _DIGIT_CARDINAL_SUFFIX_RE.sub(_digit_cardinal_suffix_replacement, text)
    return _DIGIT_CARDINAL_ATTACHED_SUFFIX_RE.sub(
        _digit_cardinal_suffix_replacement, text
    )


def _time_replacement(match: re.Match[str]) -> str:
    suffix = match.group(3)
    suffix_text = (
        f"{_CANONICAL_TIME_SUFFIX_PLACEHOLDER}{suffix.lower()}" if suffix else ""
    )
    return (
        f"{int(match.group(1))}{_CANONICAL_TIME_SEPARATOR_PLACEHOLDER}"
        f"{int(match.group(2)):02d}{suffix_text}"
    )


def _compact_time_replacement(match: re.Match[str]) -> str:
    """Canonicalize an unseparated clock and remove a leading zero from its hour."""
    suffix = match.group(3)
    suffix_text = (
        f"{_CANONICAL_TIME_SUFFIX_PLACEHOLDER}{suffix.lower()}" if suffix else ""
    )
    return (
        f"{int(match.group(1))}{_CANONICAL_TIME_SEPARATOR_PLACEHOLDER}"
        f"{match.group(2)}{suffix_text}"
    )


def _hour_prefixed_compact_time_replacement(match: re.Match[str]) -> str:
    suffix = match.group(4)
    suffix_text = (
        f"{_CANONICAL_TIME_SUFFIX_PLACEHOLDER}{suffix.lower()}" if suffix else ""
    )
    return (
        f"{match.group(1)}{int(match.group(2))}{_CANONICAL_TIME_SEPARATOR_PLACEHOLDER}"
        f"{match.group(3)}{suffix_text}"
    )


def _parse_spoken_time(
    tokens: list[str], start: int
) -> tuple[str, int] | None:
    hour_parts, hour_consumed = _collect_cardinal_parts(tokens, start)
    if not hour_parts:
        return None
    hour = _cardinal_parts_value(hour_parts)
    if hour is None or not 0 <= hour <= 23:
        return None

    minute_start = start + hour_consumed
    if minute_start < len(tokens) and tokens[minute_start] == _TIME_SEPARATOR:
        minute_start += 1
        if minute_start >= len(tokens):
            return None
        half = _split_fraction_suffix(tokens[minute_start])
        if half is not None and half[:2] == (1, 2):
            return (
                _canonical_time_token(hour, 30, half[2] or ""),
                hour_consumed + 2,
            )
        minute_parts, minute_consumed = _collect_cardinal_parts(
            tokens, minute_start
        )
        if not minute_parts:
            return None
        minute = _cardinal_parts_value(minute_parts)
        if minute is None or not 0 <= minute <= 59:
            return None
        return (
            _canonical_time_token(hour, minute, minute_parts[-1].suffix or ""),
            hour_consumed + 1 + minute_consumed,
        )

    if minute_start + 1 >= len(tokens):
        return None
    first_zero = _parse_cardinal_word(tokens[minute_start])
    second_zero = _parse_cardinal_word(tokens[minute_start + 1])
    if (
        first_zero is None
        or second_zero is None
        or first_zero.value != 0
        or second_zero.value != 0
        or first_zero.suffix is not None
    ):
        return None
    return (
        _canonical_time_token(hour, 0, second_zero.suffix or ""),
        hour_consumed + 2,
    )


def _canonical_time_token(hour: int, minute: int, suffix: str = "") -> str:
    suffix_text = (
        f"{_CANONICAL_TIME_SUFFIX_PLACEHOLDER}{suffix}" if suffix else ""
    )
    return (
        f"{hour}{_CANONICAL_TIME_SEPARATOR_PLACEHOLDER}{minute:02d}"
        f"{suffix_text}"
    )


def _parse_spoken_decimal_scale(
    tokens: list[str], start: int
) -> tuple[str, int] | None:
    whole_parts, whole_consumed = _collect_cardinal_parts(tokens, start)
    separator = start + whole_consumed
    if (
        not whole_parts
        or separator >= len(tokens)
        or tokens[separator] != _DECIMAL_SEPARATOR
    ):
        return None

    fractional_parts, fractional_consumed = _collect_cardinal_parts(
        tokens, separator + 1
    )
    if len(fractional_parts) < 2 or fractional_parts[-1].kind != "scale":
        return None
    fractional = _cardinal_parts_value(fractional_parts[:-1])
    whole = _cardinal_parts_value(whole_parts)
    if whole is None or fractional is None:
        return None

    fractional_digits = str(fractional)
    decimal = f"{whole}{_DECIMAL_SEPARATOR}{fractional_digits}"
    scale = fractional_parts[-1]
    return (
        _scaled_decimal_token(
            decimal,
            scale.value,
            scale.suffix or "",
        ),
        whole_consumed + 1 + fractional_consumed,
    )


def _fraction_replacement(match: re.Match[str]) -> str:
    return (
        f"{int(match.group(1))}{_FRACTION_SEPARATOR}"
        f"{int(match.group(2))}{match.group(3) or ''}"
    )


def _digit_ordinal_replacement(match: re.Match[str]) -> str:
    value = int(match.group(1))
    if value != 1 and re.search(r"[-‐‑–—]\s*ին", match.group(0), re.IGNORECASE):
        return f"{value}ին{match.group(2) or ''}"
    suffix = match.group(2)
    return f"{value} {_ORDINAL_MARKER}{suffix or ''}"


def _digit_cardinal_suffix_replacement(match: re.Match[str]) -> str:
    digits = match.group(1)
    canonical_digits = (
        digits if len(digits) > 1 and digits.startswith("0") else str(int(digits))
    )
    return f"{canonical_digits}{match.group(2).lower()}"


def _digit_adjectival_replacement(match: re.Match[str]) -> str:
    return f"{int(match.group(1))}{match.group(2).lower()}"


def _decimal_replacement(match: re.Match[str]) -> str:
    fractional_digits = " ".join(match.group(2))
    suffix = (match.group("suffix") or "").lstrip("-").lower()
    suffix_text = (
        f" {_CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER}{suffix}"
        if suffix
        else ""
    )
    return (
        f"{int(match.group(1))} {_DECIMAL_SEPARATOR} {fractional_digits} "
        f"{_DECIMAL_END}{suffix_text}"
    )


def _collapse_numeric_structures(tokens: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            tokens[index] != _DECIMAL_SEPARATOR
            or not output
            or not _DIGIT_TOKEN_RE.fullmatch(output[-1])
        ):
            output.append(tokens[index])
            index += 1
            continue
        if _DECIMAL_END in tokens[index:]:
            end = tokens.index(_DECIMAL_END, index + 1)
            protected = True
        else:
            end = index + 1
            while (
                end < len(tokens)
                and _DIGIT_TOKEN_RE.fullmatch(tokens[end])
                # A canonicalized scale belongs to the complete decimal, not
                # to its fractional digits: "մեկ ամբողջ հինգ միլիոն" is
                # 1.5 million, rather than the decimal 1.5000000.
                and not (
                    end > index + 1
                    and int(tokens[end]) in _SCALES.values()
                )
            ):
                end += 1
            protected = False
        digits = tokens[index + 1 : end]
        if not digits or not all(_DIGIT_TOKEN_RE.fullmatch(token) for token in digits):
            output.append(tokens[index])
            index += 1
            continue
        fractional = "".join(digits).rstrip("0") or "0"
        whole = output.pop()
        decimal = f"{whole}{_DECIMAL_SEPARATOR}{fractional}"
        next_index = end + 1 if protected else end
        if (
            protected
            and next_index < len(tokens)
            and tokens[next_index].startswith(
                _CANONICAL_DECIMAL_SUFFIX_PLACEHOLDER
            )
        ):
            decimal += tokens[next_index]
            next_index += 1
        output.append(decimal)
        index = next_index

    output = _collapse_decimal_scales(output)

    collapsed: list[str] = []
    index = 0
    while index < len(output):
        if (
            index + 2 < len(output)
            and _DIGIT_TOKEN_RE.fullmatch(output[index])
            and output[index + 1] == _TIME_SEPARATOR
            and _canonical_integer_token(output[index + 2]) is not None
        ):
            collapsed.append(
                f"{output[index]}{_TIME_SEPARATOR}{output[index + 2]}"
            )
            index += 3
            continue
        if (
            index + 2 < len(output)
            and _DIGIT_TOKEN_RE.fullmatch(output[index])
            and output[index + 1] == "ու"
        ):
            half_suffix = _half_fraction_suffix(output[index + 2])
            if half_suffix is not None:
                collapsed.append(
                    f"{output[index]}{_DECIMAL_SEPARATOR}5{half_suffix}"
                )
                index += 3
                continue
        collapsed.append(output[index])
        index += 1
    return _collapse_numeric_ranges(collapsed)


def _collapse_numeric_ranges(tokens: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens):
            start = re.fullmatch(r"(\d+)ից", tokens[index])
            end = re.fullmatch(
                r"(\d+)(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն)?",
                tokens[index + 1],
            )
            if start is not None and end is not None:
                output.append(
                    f"{start.group(1)}-{end.group(1)}{end.group(2) or ''}"
                )
                index += 2
                continue
        output.append(tokens[index])
        index += 1
    return output


def _collapse_decimal_scales(tokens: list[str]) -> list[str]:
    """Combine a canonical decimal and following Armenian scale exactly."""
    output: list[str] = []
    index = 0
    decimal_re = re.compile(rf"(\d+){_DECIMAL_SEPARATOR}(\d+)")
    while index < len(tokens):
        if index + 1 < len(tokens):
            decimal = decimal_re.fullmatch(tokens[index])
            scale = _canonical_integer_token(tokens[index + 1])
            if (
                decimal is not None
                and scale is not None
                and scale[0] in _SCALES.values()
            ):
                output.append(
                    _scaled_decimal_token(tokens[index], scale[0], scale[1])
                )
                index += 2
                continue
        output.append(tokens[index])
        index += 1
    return output


def _scaled_decimal_token(decimal: str, scale: int, suffix: str = "") -> str:
    whole, fractional = decimal.split(_DECIMAL_SEPARATOR, 1)
    numerator = int(whole + fractional) * scale
    denominator = 10 ** len(fractional)
    quotient, remainder = divmod(numerator, denominator)
    if not remainder:
        return f"{quotient}{suffix}"
    remainder_digits = str(remainder).zfill(len(fractional)).rstrip("0")
    return f"{quotient}{_DECIMAL_SEPARATOR}{remainder_digits}{suffix}"


def _canonical_integer_token(token: str) -> tuple[int, str] | None:
    match = re.fullmatch(
        r"(\d+)(ներից|ներում|ների|ներով|երին|երից|երը|ից|ում|ով|ին|ի|ը|ն)?",
        token,
    )
    if match is None:
        return None
    return int(match.group(1)), match.group(2) or ""


def _half_fraction_suffix(token: str) -> str | None:
    prefix = f"1{_FRACTION_SEPARATOR}2"
    if not token.startswith(prefix):
        return None
    suffix = token[len(prefix) :]
    return suffix if not suffix or suffix in _NUMBER_SUFFIXES else None


def _merge_grouped_integer_digits(text: str) -> str:
    for pattern in (
        _COMMA_GROUPED_INTEGER_RE,
        _DOT_GROUPED_INTEGER_RE,
        _SPACE_GROUPED_INTEGER_RE,
    ):
        text = pattern.sub(
            lambda match: (
                match.group(1)
                .replace(" ", "")
                .replace(",", "")
                .replace(".", "")
                .replace("․", "")
            ),
            text,
        )
    return text


def _parse_digit_ordinal(
    tokens: list[str], start: int
) -> tuple[list[str], int] | None:
    if (
        start + 1 >= len(tokens)
        or not _DIGIT_TOKEN_RE.fullmatch(tokens[start])
        or not tokens[start + 1].startswith(_ORDINAL_MARKER)
    ):
        return None
    suffix = tokens[start + 1][len(_ORDINAL_MARKER) :]
    if suffix and suffix not in _NUMBER_SUFFIXES:
        return None
    value = int(tokens[start])
    return [f"{value}-{_digit_ordinal_suffix(value)}{suffix}"], 2


def _parse_spoken_fraction(
    tokens: list[str], start: int
) -> tuple[list[str], int] | None:
    direct = _split_fraction_suffix(tokens[start])
    if direct is not None:
        numerator, denominator, suffix = direct
        if (numerator, denominator) == (1, 2):
            return None
        output = [
            f"{numerator}{_FRACTION_SEPARATOR}{denominator}{suffix or ''}"
        ]
        return output, 1

    parts, consumed = _collect_cardinal_parts(tokens, start)
    if not parts or start + consumed >= len(tokens):
        return None
    fraction = _split_fraction_suffix(tokens[start + consumed])
    if fraction is None:
        return None
    _, denominator, suffix = fraction
    if denominator == 2:
        return None
    numerator = _cardinal_parts_value(parts)
    if numerator is None:
        return None
    output = [
        f"{numerator}{_FRACTION_SEPARATOR}{denominator}{suffix or ''}"
    ]
    return output, consumed + 1


def _parse_compound_half(token: str) -> str | None:
    if token in _COMPOUND_HALF_WORDS:
        return f"{_COMPOUND_HALF_WORDS[token]}{_DECIMAL_SEPARATOR}5"
    if token.endswith("ուկես"):
        cardinal = _parse_cardinal_word(token[: -len("ուկես")])
        if cardinal is not None and cardinal.suffix is None:
            return f"{cardinal.value}{_DECIMAL_SEPARATOR}5"
    for suffix in _NUMBER_SUFFIXES:
        if not token.endswith(suffix):
            continue
        base = token[: -len(suffix)]
        if base in _COMPOUND_HALF_WORDS:
            return (
                f"{_COMPOUND_HALF_WORDS[base]}{_DECIMAL_SEPARATOR}5{suffix}"
            )
        if base.endswith("ուկես"):
            cardinal = _parse_cardinal_word(base[: -len("ուկես")])
            if cardinal is not None and cardinal.suffix is None:
                return f"{cardinal.value}{_DECIMAL_SEPARATOR}5{suffix}"
    return None


def _split_fraction_suffix(token: str) -> tuple[int, int, str | None] | None:
    if token in _FRACTION_WORDS:
        numerator, denominator = _FRACTION_WORDS[token]
        return numerator, denominator, None
    for suffix in _NUMBER_SUFFIXES:
        if token.endswith(suffix):
            base = token[: -len(suffix)]
            if base in _FRACTION_WORDS:
                numerator, denominator = _FRACTION_WORDS[base]
                return numerator, denominator, suffix
    return None


def _parse_spoken_ordinal_sequence(
    tokens: list[str], start: int
) -> tuple[list[str], int] | None:
    direct = _parse_ordinal_word(tokens[start])
    if direct is not None:
        part, suffix = direct
        output = [
            f"{part.value}-{_digit_ordinal_suffix(part.value)}{suffix or ''}"
        ]
        return output, 1

    parts, consumed = _collect_cardinal_parts(tokens, start)
    if not parts or start + consumed >= len(tokens):
        return None
    ordinal = _parse_ordinal_word(tokens[start + consumed])
    if ordinal is None:
        return None
    ordinal_part, suffix = ordinal
    if not _valid_cardinal_transition(parts[-1], ordinal_part):
        return None
    value = _cardinal_parts_value([*parts, ordinal_part])
    if value is None:
        return None
    output = [f"{value}-{_digit_ordinal_suffix(value)}{suffix or ''}"]
    return output, consumed + 1


def _digit_ordinal_suffix(value: int) -> str:
    return "ին" if value == 1 else "րդ"


def _parse_ordinal_word(token: str) -> tuple[_NumberPart, str | None] | None:
    if token in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[token], None
    for suffix in _NUMBER_SUFFIXES:
        if token.endswith(suffix):
            base = token[: -len(suffix)]
            if base in _ORDINAL_WORDS:
                return _ORDINAL_WORDS[base], suffix
    return None


def _parse_cardinal_sequence(
    tokens: list[str], start: int
) -> tuple[list[str], int] | None:
    parts, consumed = _collect_cardinal_parts(tokens, start)
    if not parts:
        return None
    value = _cardinal_parts_value(parts)
    if value is None:
        return None
    output = [f"{value}{parts[-1].suffix or ''}"]
    return output, consumed


def _collect_cardinal_parts(
    tokens: list[str], start: int
) -> tuple[list[_NumberPart], int]:
    parts: list[_NumberPart] = []
    index = start
    while index < len(tokens):
        part = _parse_cardinal_word(tokens[index])
        if part is None:
            break
        if parts and not _valid_cardinal_transition(parts[-1], part):
            break
        parts.append(part)
        index += 1
        if part.suffix is not None:
            break
    return parts, index - start


def _parse_cardinal_word(token: str) -> _NumberPart | None:
    if _DIGIT_TOKEN_RE.fullmatch(token):
        if len(token) > 1 and token.startswith("0"):
            return None
        return _NumberPart(int(token), "small")

    # The final letter in the standalone cardinal ``տասը`` is part of its
    # lexical form, not a grammatical suffix that should survive conversion.
    if token == "տասը":
        return _NumberPart(10, "small")

    irregular = _IRREGULAR_INFLECTED_CARDINALS.get(token)
    if irregular is not None:
        base, suffix = irregular
        return _NumberPart(_SMALL_CARDINALS[base], "small", suffix)

    base, suffix = _split_cardinal_suffix(token)
    if base in _SMALL_CARDINALS:
        return _NumberPart(_SMALL_CARDINALS[base], "small", suffix)
    if base == "հարյուր":
        return _NumberPart(100, "hundred", suffix)
    if base in _SCALES:
        return _NumberPart(_SCALES[base], "scale", suffix)
    return None


def _split_cardinal_suffix(token: str) -> tuple[str, str | None]:
    if token in _CARDINAL_WORDS:
        return token, None
    for suffix in _NUMBER_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            base = token[: -len(suffix)]
            if base in _CARDINAL_WORDS:
                return base, suffix
    return token, None


def _valid_cardinal_transition(previous: _NumberPart, current: _NumberPart) -> bool:
    if previous.suffix is not None:
        return False
    if current.kind == "hundred":
        return (
            previous.kind == "scale"
            or previous.kind == "small"
            and 1 <= previous.value <= 9
        )
    if current.kind == "scale":
        return previous.kind in {"small", "hundred"}
    if current.kind != "small":
        return False
    if previous.kind == "hundred":
        return current.value < 100
    if previous.kind == "scale":
        return current.value < previous.value
    return (
        previous.kind == "small"
        and previous.value in _TENS.values()
        and 1 <= current.value <= 9
    )


def _cardinal_parts_value(parts: list[_NumberPart]) -> int | None:
    total = 0
    current = 0
    last_scale = float("inf")
    for part in parts:
        if part.kind == "small":
            current += part.value
        elif part.kind == "hundred":
            current = max(current, 1) * 100
        elif part.kind == "scale":
            if part.value >= last_scale:
                return None
            total += max(current, 1) * part.value
            current = 0
            last_scale = part.value
        else:
            return None
    return total + current
