from dataclasses import dataclass
from pathlib import Path

import yaml

from asap.config import Settings, get_settings

EXPERIMENTS_FILE = Path("config/experiments.yml")


@dataclass(frozen=True)
class Variant:
    name: str
    base_model: str
    learning_rate: float
    epochs: int


def load_variants(path: Path | None = None) -> dict[str, Variant]:
    path = path or EXPERIMENTS_FILE
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        name: Variant(name=name, **spec)
        for name, spec in (raw.get("variants") or {}).items()
    }


def variant_out_dir(name: str, *, settings: Settings | None = None) -> Path:
    cfg = settings or get_settings()
    return cfg.paths.experiments_dir / name


def select_best(scores: dict[str, float], *, greater_is_better: bool) -> str:
    if not scores:
        raise ValueError("no variant scores to choose best model from")
    pick = max if greater_is_better else min
    return pick(scores, key=lambda name: scores[name])
