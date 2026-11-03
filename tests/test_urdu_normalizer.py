"""Regression cases for the Urdu benchmark scoring normalizer."""
from normalizer.urdu_normalizer import UrduTextNormalizer


def test_urdu_arabic_letter_variants():
    n = UrduTextNormalizer()
    assert n("كتاب میں یہ كيا ہے؟") == "کتاب میں یہ کیا ہے"
    assert n("كراچي اور کراچی") == "کراچی اور کراچی"


def test_urdu_diacritics_punctuation_and_whitespace():
    n = UrduTextNormalizer()
    assert n("  کِتاب،   حُسَین!\n") == "کتاب حسین"
    assert n("سَلامـ! دنیا۔") == "سلام دنیا"


def test_urdu_preserves_distinct_letters():
    n = UrduTextNormalizer()
    assert n("یہ رہے ہیں") == "یہ رہے ہیں"
    assert n("گھر میں گئے") == "گھر میں گئے"
    assert n("آئے، آئی") == "آئے آئی"


def test_urdu_digit_scripts_and_empty():
    n = UrduTextNormalizer()
    assert n("سن ۲۰۲۶، ٢٠٢٦ اور 2026") == "سن 2026 2026 اور 2026"
    assert n("") == ""
    assert n(None) == ""


def test_urdu_idempotent_and_script_preservation():
    n = UrduTextNormalizer()
    value = "  اسلام‌آباد: ۳۰۰ روپے اور 20 USD!  "
    assert n(n(value)) == n(value)
    assert n(value) == "اسلام آباد 300 روپے اور 20 usd"


def test_urdu_retains_hamza_and_distinguishes_bari_yeh():
    n = UrduTextNormalizer()
    assert n("مسئلہ اور گئے") == "مسئلہ اور گئے"
