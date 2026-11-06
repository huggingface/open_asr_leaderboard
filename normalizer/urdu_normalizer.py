"""Conservative Urdu orthographic normalizer for ASR WER scoring.

Decisions deliberately avoid lexical rewriting, aggressive hamza conversion,
translation, or converting numeric values into guesses about their spoken forms.
The identical transform must be applied to references and hypotheses.
"""

import re
import unicodedata

# Arabic and Persian code points with stable Urdu equivalents. Bari yeh (ے)
# and small yeh (ی) remain distinct; letter heh (ہ) and do-chashmi heh (ھ)
# also remain distinct.
_LETTER_MAP = str.maketrans({
    "ك": "ک",  # Arabic Kaf -> Urdu Keheh
    "ي": "ی",  # Arabic Yeh -> Farsi Yeh
    "ى": "ی",  # Alef Maksura -> Yeh (OCR/transcription variant)
    "ه": "ہ",  # Arabic Heh -> Urdu Heh
})

# Do not remove hamza-above/below: in some Unicode sequences these marks
# distinguish Urdu lexical spellings where a precomposed character is absent.
_KEEP_COMBINING = {"\u0654", "\u0655"}
_CONTROL_AS_SPACE = {"\u200c", "\u200d", "\u2060"}


class UrduTextNormalizer:
    """Return normalized, whitespace-separated Urdu text for word-level WER."""

    def __call__(self, value: str) -> str:
        if value is None:
            return ""
        text = unicodedata.normalize("NFKC", str(value)).lower().translate(_LETTER_MAP)
        result = []
        for char in text:
            if char in _CONTROL_AS_SPACE:
                result.append(" ")
                continue
            if char == "\u0640":  # Kashida/tatweel is only ornamental
                continue
            if char in _KEEP_COMBINING:
                result.append(char)
                continue
            category = unicodedata.category(char)
            if category.startswith("M"):  # vowels and Quranic annotation marks
                continue
            if category.startswith("P") or category.startswith("S") or category.startswith("C"):
                result.append(" ")
                continue
            if char.isdecimal():
                result.append(str(unicodedata.decimal(char)))
            else:
                result.append(char)
        return re.sub(r"\s+", " ", "".join(result)).strip()
