import pytest

from asap.preprocessing import normalize


def test_collapses_runs_of_three_or_more():
    assert normalize("رااااائع") == "رائع"


def test_preserves_legitimate_doubled_letters():
    assert normalize("اللغة") == "اللغه"


def test_removes_tatweel():
    assert normalize("كــــــتاب") == "كتاب"


def test_removes_diacritics():
    assert normalize("مُحَمَّد") == "محمد"


def test_unifies_alef_forms():
    assert normalize("أحمد إبراهيم آمن") == "احمد ابراهيم امن"


def test_unifies_ta_marbuta_and_alef_maqsura():
    assert normalize("جميلة على") == "جميله علي"


@pytest.mark.parametrize(
    "negator", ["لا", "لم", "لن", "ليس", "ما"]
)
def test_retains_negation_particles(negator):
    assert negator in normalize(f"{negator} أنصح بهذا المنتج")


def test_retains_punctuation():
    assert "!" in normalize("المنتج رائع!")


def test_strips_urls_and_mentions():
    out = normalize("شوف https://example.com/x و @user المنتج")
    assert "example.com" not in out
    assert "@user" not in out
    assert "المنتج" in out


def test_squashes_whitespace():
    assert normalize("  المنتج    رائع  ") == "المنتج رائع"


@pytest.mark.parametrize("bad", [None, 123, float("nan"), []])
def test_non_string_returns_empty(bad):
    assert normalize(bad) == ""


def test_is_idempotent():
    raw = "رااااائع جـــداً!! أنا لا أصدق 😍 https://x.co/1"
    assert normalize(normalize(raw)) == normalize(raw)