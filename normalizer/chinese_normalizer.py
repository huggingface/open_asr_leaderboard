import re
import unicodedata

import regex

from .normalizer import remove_symbols_keep_marks

# Vulgar fractions such as "¾", which NFKC would otherwise merge with a
# preceding digit ("29¾" -> "293⁄4").
VULGAR_FRACTION_RE = re.compile(r"(\d?)([¼-¾⅐-⅞])")
BRACKETS_RE = re.compile(r"[<\[][^>\]]*[>\]]")
SPACE_NEXT_TO_HAN_RE = regex.compile(r"(?<=\p{Han})\s+|\s+(?=\p{Han})")


def _split_vulgar_fraction(match):
    fraction = unicodedata.normalize("NFKC", match.group(2)).replace("⁄", "/")
    return f"{match.group(1)} {fraction}" if match.group(1) else fraction


class ChineseTextNormalizer:
    """Mandarin Chinese normalizer, applied to references and predictions alike.

    Chinese is scored with CER (see CER_LANGUAGES in eval_utils.py), which splits
    the normalized text into characters. Steps:
      1. split vulgar fractions off the digit before them ("29¾" -> "29 3/4")
      2. NFKC (full-width -> half-width)
      3. Traditional -> Simplified characters (OpenCC t2s)
      4. lower case
      5. remove words between [] and <> (e.g. "[音乐]"); replace ( and ) with
         a space, keeping the text between them
      6. remove whitespace next to Chinese characters ("2007 年" -> "2007年")
      7. Arabic numbers -> Chinese numerals with wetext ("2007年" -> "二零零七年"),
         and "〇" -> "零", the zero wetext writes ("二〇〇七年" -> "二零零七年")
      8. replace symbols and punctuation with a space, collapse whitespace

    opencc and wetext are imported on first use: data_utils builds this
    normalizer at import time for every language, and not every evaluation
    image has them installed.
    """

    def __init__(self):
        self._t2s = None
        self._number_normalizer = None

    def _load(self):
        # deferred imports: only needed for Chinese
        import opencc
        from wetext import Normalizer

        self._t2s = opencc.OpenCC("t2s")
        # Numbers and symbols only (no t2s, punctuation or interjection removal)
        self._number_normalizer = Normalizer(lang="zh", operator="tn")

    def __call__(self, s: str) -> str:
        if self._number_normalizer is None:
            self._load()
        s = VULGAR_FRACTION_RE.sub(_split_vulgar_fraction, s)
        s = unicodedata.normalize("NFKC", s)
        s = self._t2s.convert(s)
        s = s.lower()
        s = BRACKETS_RE.sub("", s)
        s = re.sub(r"[()]", " ", s)
        s = SPACE_NEXT_TO_HAN_RE.sub("", s)
        s = self._number_normalizer.normalize(s)
        s = s.replace("〇", "零")
        s = remove_symbols_keep_marks(s)
        s = regex.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
