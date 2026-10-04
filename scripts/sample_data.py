from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from asap.config import Settings, get_settings


def build_sample(
    out_path: Path | None = None,
    n_rows: int = 500,
    *,
    settings: Settings | None = None,
) -> Path:
    """Sample `n_rows` reviews from the test split into a JSONL fixture."""
    cfg = settings or get_settings()
    out_path = out_path if out_path is not None else cfg.paths.bench_sample

    parquet = cfg.paths.processed_dir / "test.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(
            f"no test split at {parquet}; build it first with "
            "`python -m asap.data.build`"
        )

    frame = pd.read_parquet(parquet)
    if "text" not in frame.columns:
        raise KeyError(
            f"{parquet} has no 'text' column; found {list(frame.columns)}. "
            "The processed split stores normalized text under 'text'."
        )

    if len(frame) < n_rows:
        warnings.warn(
            f"{parquet} has {len(frame)} rows, fewer than the requested "
            f"{n_rows}; using all of them",
            stacklevel=2,
        )
        sample = frame
    else:
        sample = frame.sample(n=n_rows, random_state=cfg.project["seed"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for text in sample["text"]:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the load-test fixture.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-rows", type=int, default=500)
    args = ap.parse_args()

    path = build_sample(args.out, args.n_rows)
    rows = sum(1 for _ in path.open(encoding="utf-8"))
    print(f"{path} ({rows:,} rows)")


if __name__ == "__main__":
    main()
