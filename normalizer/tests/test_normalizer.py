"""Synthetic regressions for bracketed speech and existing language routing.

Run from the repository root: python -m unittest discover -s normalizer/tests -v
"""

import unittest
from unittest.mock import patch

from normalizer import BasicMultilingualTextNormalizer, EnglishTextNormalizer
from normalizer.normalizer import BasicTextNormalizer


BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("<", ">"), ("（", "）"), ("［", "］"), ("＜", "＞"))
NON_SPEECH_TAGS = ("laugh", "laughs", "laughing", "laughter", "cough", "coughs", "coughing", "unk", "music")


class AnnotationNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizers = (
            BasicTextNormalizer(),
            BasicTextNormalizer(remove_diacritics=True),
            BasicMultilingualTextNormalizer(),
            BasicMultilingualTextNormalizer(remove_diacritics=False),
            EnglishTextNormalizer(),
        )

    def assert_normalized(self, source, expected):
        for index, normalizer in enumerate(self.normalizers):
            with self.subTest(normalizer=index, source=source):
                self.assertEqual(normalizer(source).strip(), expected)

    def test_preserves_spoken_words_acronyms_and_quantities(self):
        self.assert_normalized("The agency (NASA) sent [spoken words].", "the agency nasa sent spoken words")
        for opening, closing in BRACKET_PAIRS:
            with self.subTest(opening=opening):
                self.assert_normalized(f"before{opening}ordinary words{closing}after", "before ordinary words after")
        self.assertEqual(EnglishTextNormalizer()("A distance of (1,040 kilometers).").strip(), "a distance of 1040 kilometers")
        self.assertEqual(BasicMultilingualTextNormalizer()("Une distance de (1 040 kilomètres)."), "une distance de 1 040 kilometres")

    def test_removes_only_exact_non_speech_tags(self):
        for tag in NON_SPEECH_TAGS:
            for opening, closing in BRACKET_PAIRS:
                with self.subTest(tag=tag, opening=opening):
                    self.assert_normalized(f"before{opening} \t{tag.upper()}\n {closing}after", "before after")

    def test_preserves_longer_phrases_and_unenclosed_tag_words(self):
        self.assert_normalized("Try (cough medicine) and [laughing out loud].", "try cough medicine and laughing out loud")
        self.assert_normalized("Study (music theory) with <unknown words>.", "study music theory with unknown words")
        self.assert_normalized("laugh cough laughing music unknown", "laugh cough laughing music unknown")

    def test_handles_nested_tags_and_spoken_parentheticals(self):
        self.assert_normalized("before (spoken [cough] words) after", "before spoken words after")
        self.assert_normalized("before [spoken （LAUGH） words] after", "before spoken words after")
        self.assert_normalized("before (laugh [cough]) after", "before after")
        self.assert_normalized("before (outer (inner words) end) after", "before outer inner words end after")

    def test_requires_matching_complete_delimiters(self):
        for malformed, word in (("(laugh]", "laugh"), ("[cough)", "cough"), ("<unk]", "unk"), ("(laugh", "laugh"), ("cough)", "cough"), ("（laugh)", "laugh"), ("(cough）", "cough")):
            with self.subTest(malformed=malformed):
                self.assert_normalized(f"before {malformed} after", f"before {word} after")

    def test_empty_and_adjacent_enclosures_keep_word_boundaries(self):
        self.assert_normalized("before()[cough]<unk>after", "before after")
        self.assert_normalized("before(alpha)[beta]after", "before alpha beta after")

    def test_annotation_filter_is_idempotent(self):
        from normalizer.normalizer import remove_non_speech_annotations

        source = "(ordinary words) [cough medicine] <music theory>"
        self.assertEqual(remove_non_speech_annotations(source), source)
        cleaned = remove_non_speech_annotations("before (spoken [cough] words) after (laugh)")
        self.assertEqual(remove_non_speech_annotations(cleaned), cleaned)

    def test_existing_english_rules(self):
        normalizer = EnglishTextNormalizer()
        for text, expected in (("Uh, we can't meet at 2:00 AM.", "we can not meet at 2 am"), ("The colour of the wi-fi router.", "the color of the wifi router"), ("Pay twenty dollars.", "pay $20")):
            with self.subTest(text=text):
                self.assertEqual(normalizer(text).strip(), expected)

    def test_existing_unicode_and_split_letter_rules(self):
        self.assertEqual(BasicMultilingualTextNormalizer(remove_diacritics=False)("हिन्दी (किताब)"), "हिन्दी किताब")
        self.assertEqual(BasicMultilingualTextNormalizer(remove_diacritics=False)("café [déjà vu]"), "café déjà vu")
        self.assertEqual(BasicTextNormalizer(split_letters=True)("ab(cd)").split(), list("abcd"))


class ScoringRegressionTests(unittest.TestCase):
    def test_bracketed_reference_and_prediction_score_symmetrically(self):
        from normalizer.eval_utils import score_manifest

        plain = "they said ordinary words today"
        enclosed = "they said (ordinary words) today"
        for reference, prediction in ((enclosed, plain), (plain, enclosed)):
            with self.subTest(reference=reference):
                score = score_manifest([{"text": reference, "pred_text": prediction, "duration": 2.0, "time": 0.5}], num_workers=1)
                self.assertEqual((score["wer"], score["ins"], score["del"], score["sub"]), (0.0, 0, 0, 0))
                self.assertEqual(score["rtfx"], 4.0)

    def test_multilingual_numbers_still_use_language_routing(self):
        from normalizer.eval_utils import _normalize

        for language in ("de", "fr", "es", "it", "pt", "nl"):
            with self.subTest(language=language):
                plain = _normalize("12", language)
                self.assertNotEqual(plain, "12")
                self.assertEqual(_normalize("(12)", language), plain)

    def test_armenian_already_preserves_enclosed_words_and_numbers(self):
        from normalizer.eval_utils import _normalize

        for plain in ("բարեւ աշխարհ", "12"):
            for opening, closing in BRACKET_PAIRS:
                with self.subTest(plain=plain, opening=opening):
                    self.assertEqual(_normalize(f"{opening}{plain}{closing}", "hy"), _normalize(plain, "hy"))

    def test_hindi_scoring_keeps_its_own_oiwer_normalization(self):
        from normalizer import eval_utils

        manifest = [{"text": "हिन्दी (किताब)", "pred_text": "हिन्दी किताब", "duration": None, "time": None}]
        with patch.object(eval_utils, "score_oiwer", return_value=(0.0, 0, 0, 0)) as oiwer, patch.object(eval_utils, "_normalize", side_effect=AssertionError("Hindi uses voi_oiwer")):
            self.assertEqual(eval_utils.score_manifest(manifest, language="hi")["wer"], 0.0)
        oiwer.assert_called_once_with(manifest, "hindi")


if __name__ == "__main__":
    unittest.main()
