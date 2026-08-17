import json
import os
import tempfile
import unittest

from ref_rendering_utils import (
    REPORTED_CLASSES,
    SCORED_CLASSES,
    classify,
    comparable,
    flagged_spans_for,
    keep_flagged_span,
    score_pairs,
)
from score_ref_rendering import discover_datasets, read_manifest


class RenderingClassificationTests(unittest.TestCase):
    def test_only_clean_classes_are_reported_and_scored(self):
        self.assertEqual(REPORTED_CLASSES, ("spelling", "acronym", "number"))
        self.assertEqual(SCORED_CLASSES, {"spelling", "acronym", "number"})

    def test_real_rendering_classes(self):
        self.assertEqual(classify("colour", "color"), "spelling")
        self.assertEqual(classify("T. V.", "tv"), "acronym")
        self.assertEqual(classify("forty seven", "47"), "number")

    def test_ambiguous_abbreviation_is_excluded(self):
        span = ("gen", "general", "abbrev", 0, 0, 0, 1)
        self.assertFalse(keep_flagged_span(span))

    def test_number_filter_rejects_merge_and_large_number(self):
        merge = ("one one", "11", "number", 0, 1, 0, 1)
        year = ("nineteen eighty four", "1984", "number", 0, 2, 0, 1)
        self.assertFalse(keep_flagged_span(merge))
        self.assertFalse(keep_flagged_span(year))


class RenderingScoringTests(unittest.TestCase):
    def test_case_and_edge_punctuation_do_not_change_agreement(self):
        self.assertEqual(comparable("COLOUR,"), comparable("colour"))
        result = score_pairs([("We value colour.", "we value COLOUR")])
        self.assertEqual(result["scored"], [1, 1])

    def test_number_hyphenation_is_not_a_disagreement(self):
        result = score_pairs([("There were fifty-two people.", "there were fifty two people")])
        self.assertEqual(result["scored"], [1, 1])
        self.assertEqual(result["by_class"]["number"], [1, 1])

    def test_unicode_number_hyphen_is_accepted(self):
        span = ("fifty‑two", "52", "number", 0, 0, 0, 1)
        self.assertTrue(keep_flagged_span(span))
        self.assertEqual(comparable("fifty‑two", "number"), "fifty two")

    def test_digits_versus_words_is_a_disagreement(self):
        result = score_pairs([("There were fifty-two people.", "there were 52 people")])
        self.assertEqual(result["scored"], [0, 1])

    def test_pointed_acronym_versus_compact_is_a_disagreement(self):
        result = score_pairs([("Turn on the T. V.", "turn on the TV")])
        self.assertEqual(result["scored"], [0, 1])
        self.assertEqual(result["by_class"]["acronym"], [0, 1])

    def test_wrong_words_are_ineligible(self):
        result = score_pairs([("We value colour.", "we value hue")])
        self.assertEqual(result["scored"], [0, 0])

    def test_abbreviations_never_reach_scoring(self):
        spans, _ = flagged_spans_for("The gen was discussed.")
        self.assertEqual(spans, ())
        result = score_pairs([("The gen was discussed.", "the general was discussed")])
        self.assertEqual(result["scored"], [0, 0])


class RenderingInputTests(unittest.TestCase):
    def test_manifest_reader_skips_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "a", "pred_text": "a"}) + "\n")
                f.write("not json\n")
                f.write("[]\n")
            self.assertEqual(read_manifest(path), [{"text": "a", "pred_text": "a"}])

    def test_derived_voxpopuli_set_is_not_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "voxpopuli_cleaned_aa_test"))
            path = os.path.join(tmp, "voxpopuli_cleaned_aa_test", "model.jsonl")
            open(path, "w", encoding="utf-8").close()
            self.assertEqual(discover_datasets(tmp), [])


if __name__ == "__main__":
    unittest.main()
