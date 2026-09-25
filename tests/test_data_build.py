import pandas as pd
import pytest

from asap.data.build import build_splits


@pytest.fixture
def raw_csv(tmp_path):
    rows = []
    for i in range(400):
        label = i % 2
        content = "هذا المنتج جيد جدا ولا اندم على شرائه ابدا" if i < 20 else (
            f"مراجعة رقم {i} عن المنتج وهي مراجعة مفصلة وطويلة بما يكفي للاختبار"
        )
        rows.append({"label": label, "content": content})
    p = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_no_normalized_text_leaks_across_splits(raw_csv, tmp_path):
    paths = build_splits(raw_csv, tmp_path / "out")
    frames = {k: pd.read_parquet(v) for k, v in paths.items()}
    tr, va, te = (set(frames[k]["text"]) for k in ("train", "val", "test"))
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()


def test_deduplicates_on_normalized_text(raw_csv, tmp_path):
    paths = build_splits(raw_csv, tmp_path / "out")
    all_text = pd.concat([pd.read_parquet(p) for p in paths.values()])["text"]
    assert all_text.duplicated().sum() == 0


def test_labels_are_stratified(raw_csv, tmp_path):
    paths = build_splits(raw_csv, tmp_path / "out")
    ratios = [pd.read_parquet(p)["label"].mean() for p in paths.values()]
    assert max(ratios) - min(ratios) < 0.10


def test_text_column_is_normalized(raw_csv, tmp_path):
    from asap.preprocessing import normalize

    paths = build_splits(raw_csv, tmp_path / "out")
    df = pd.read_parquet(paths["train"])
    assert (df["text"] == df["text"].map(normalize)).all()


def test_token_length_filter_applied(raw_csv, tmp_path):
    paths = build_splits(raw_csv, tmp_path / "out", min_tokens=6, max_tokens=130)
    df = pd.read_parquet(paths["train"])
    counts = df["text"].str.split().str.len()
    assert counts.between(6, 130).all()