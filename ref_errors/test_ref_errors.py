import json
import os
import tempfile
import unittest

from ref_error_utils import RefEdit, char_distance, find_ref_errors, ref_error_agreement
from score_ref_errors import collect_hypotheses, read_manifest


class VerdictTest(unittest.TestCase):
    def test_replacement_verdicts(self):
        edits = find_ref_errors(
            "before bad words after context".split(),
            "before good after context",
            {
                "ref": "before bad words after context",
                "ref_with_extra": "before bad words good after context",
                "consensus": "before good after context",
                "third": "before different after context",
                "partial_ref": "before bad after context",
                "partial_ref_with_correction": "before bad good after context",
            },
        )

        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].text, "bad words")
        self.assertEqual(edits[0].corrected_text, "good")
        self.assertEqual(
            edits[0].verdict,
            {
                "ref": "ref",
                "ref_with_extra": "ref",
                "consensus": "consensus",
                "third": None,
                "partial_ref": None,
                "partial_ref_with_correction": None,
            },
        )

    def test_partial_retention_is_not_a_corrected_deletion(self):
        edits = find_ref_errors(
            "before bad words after".split(),
            "before after",
            {
                "ref": "before bad words after",
                "consensus": "before after",
                "partial_ref": "before bad after",
            },
        )
        self.assertEqual(
            edits[0].verdict,
            {"ref": "ref", "consensus": "consensus", "partial_ref": None},
        )

    def test_boundary_insertion_verdicts(self):
        start = find_ref_errors(
            "alpha beta gamma".split(),
            "new one alpha beta gamma",
            {
                "ref": "alpha beta gamma",
                "consensus": "new one alpha beta gamma",
                "consensus_with_extra": "junk new one alpha beta gamma",
                "third": "junk alpha beta gamma",
            },
        )
        self.assertEqual(len(start), 1)
        self.assertEqual(
            start[0].verdict,
            {
                "ref": "ref",
                "consensus": "consensus",
                "consensus_with_extra": "consensus",
                "third": None,
            },
        )

        end = find_ref_errors(
            "alpha beta gamma".split(),
            "alpha beta gamma new one",
            {
                "ref": "alpha beta gamma",
                "consensus": "alpha beta gamma new one",
                "consensus_with_extra": "alpha beta gamma new one junk",
                "third": "alpha beta gamma junk",
            },
        )
        self.assertEqual(len(end), 1)
        self.assertEqual(end[0].verdict, start[0].verdict)

    def test_competence_and_length_gates(self):
        edits = find_ref_errors(
            "one two bad four five six".split(),
            "one two good four five six",
            {
                "competent": "one two bad four five six",
                "too_distant": "bad",
                "runaway": "one two bad four five six " * 4,
            },
        )
        self.assertEqual(edits[0].verdict["competent"], "ref")
        self.assertIsNone(edits[0].verdict["too_distant"])
        self.assertIsNone(edits[0].verdict["runaway"])

    def test_spacing_is_not_a_reference_error(self):
        self.assertEqual(char_distance("anti corruption", "anticorruption"), 0.0)
        self.assertEqual(
            find_ref_errors(
                "before anti corruption after".split(),
                "before anticorruption after",
                {"model": "before anti corruption after"},
            ),
            [],
        )


class AggregateTest(unittest.TestCase):
    def test_counts_spans_and_clips_without_intervals(self):
        edits = [
            RefEdit("delete", "start", ["a"], [0], verdict={"m": "ref"}, clip_key="c1"),
            RefEdit("delete", "middle", ["b", "c"], [1, 2], verdict={"m": "consensus"}, clip_key="c1"),
            RefEdit("insert", "end", ["d"], [3], verdict={"m": "ref"}, clip_key="c2"),
            RefEdit("delete", "end", ["e"], [4], verdict={"m": None}, clip_key="c3"),
        ]

        self.assertEqual(
            ref_error_agreement(edits)["m"],
            {"rate": 2 / 3, "n_ref": 2, "n_eligible": 3, "n_clips": 2},
        )


class ManifestTest(unittest.TestCase):
    def test_malformed_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "manifest.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json\n")
                f.write("[]\n")
                f.write(json.dumps({"audio_filepath": "a", "text": "ref"}) + "\n")
                f.write(
                    json.dumps(
                        {"audio_filepath": "b", "text": "reference", "pred_text": "prediction"}
                    )
                    + "\n"
                )

            self.assertEqual(
                read_manifest(path),
                [{"audio_filepath": "b", "text": "reference", "pred_text": "prediction"}],
            )

    def test_opaque_keys_require_matching_reference_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.jsonl")
            bad = os.path.join(tmp, "bad.jsonl")
            keyed = os.path.join(tmp, "keyed.jsonl")

            def write(path, rows):
                with open(path, "w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            write(
                keyed,
                [
                    {"audio_filepath": "k1", "text": "r1", "pred_text": "p1"},
                    {"audio_filepath": "k2", "text": "r2", "pred_text": "p2"},
                ],
            )
            write(
                good,
                [
                    {"audio_filepath": "sample_0", "text": "r1", "pred_text": "g1"},
                    {"audio_filepath": "sample_1", "text": "r2", "pred_text": "g2"},
                ],
            )
            write(
                bad,
                [
                    {"audio_filepath": "sample_0", "text": "r2", "pred_text": "b1"},
                    {"audio_filepath": "sample_1", "text": "r1", "pred_text": "b2"},
                ],
            )

            hypotheses, skipped = collect_hypotheses(
                {"keyed": keyed, "good": good, "bad": bad}, ["k1", "k2"]
            )
            self.assertEqual(hypotheses["good"], {"k1": "g1", "k2": "g2"})
            self.assertNotIn("bad", hypotheses)
            self.assertEqual(skipped, [("bad", "2/2 references disagree by row order")])


if __name__ == "__main__":
    unittest.main()
