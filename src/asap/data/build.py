from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from asap.config import Settings, get_settings
from asap.preprocessing import normalize


def build_splits(
    raw_csv: Path | None = None,
    out_dir: Path | None = None,
    *,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Path]:

    cfg = settings or get_settings()

    raw_csv = raw_csv if raw_csv is not None else cfg.paths.raw_csv
    out_dir = out_dir if out_dir is not None else cfg.paths.processed_dir
    min_tokens = min_tokens if min_tokens is not None else cfg.data.min_tokens
    max_tokens = max_tokens if max_tokens is not None else cfg.data.max_tokens
    seed = seed if seed is not None else cfg.project["seed"]

    label_col = cfg.data.label_column
    text_col = cfg.data.text_column

    df = pd.read_csv(raw_csv)
    df = df[[label_col, text_col]].dropna()

    df["text"] = df[text_col].map(normalize)
    df = df[df["text"].str.len() > 0]
    if cfg.data.dedup_on_normalized_text:
        df = df.drop_duplicates(subset="text", keep="first")

    n_tokens = df["text"].str.split().str.len()
    df = df[n_tokens.between(min_tokens, max_tokens)]

    df = df[["text", label_col]].rename(columns={label_col: "label"})
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype("int64")

    stratify_on = df["label"] if cfg.data.stratify else None
    holdout = cfg.data.val_fraction + cfg.data.test_fraction
    train, rest = train_test_split(
        df, test_size=holdout, stratify=stratify_on, random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=cfg.data.test_fraction / holdout,
        stratify=rest["label"] if cfg.data.stratify else None,
        random_state=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        p = out_dir / f"{name}.parquet"
        frame.reset_index(drop=True).to_parquet(p, index=False)
        paths[name] = p
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for name, path in build_splits(args.raw, args.out).items():
        print(f"{name}: {path} ({len(pd.read_parquet(path)):,} rows)")