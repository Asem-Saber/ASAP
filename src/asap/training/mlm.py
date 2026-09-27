import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from asap.config import Settings, get_settings
from asap.preprocessing import normalize

logger = logging.getLogger(__name__)


def load_corpus(
    raw_csv: Path,
    text_column: str,
    *,
    max_rows: int | None = None,
    normalize_text: bool = True,
) -> list[str]:
    """Read the review text for pretraining, optionally normalized."""
    frame = pd.read_csv(raw_csv, nrows=max_rows)
    if text_column not in frame.columns:
        raise KeyError(f"{text_column!r} not in {raw_csv}; has {list(frame.columns)}")

    series = frame[text_column].dropna().astype("string")
    if normalize_text:
        series = series.map(normalize)

    texts = [t for t in series.tolist() if t and t.strip()]
    if not texts:
        raise ValueError(f"no usable text in {raw_csv}")
    return texts


def train_mlm(
    raw_csv: Path | None = None,
    out_dir: Path | None = None,
    *,
    base_model: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    max_length: int | None = None,
    use_fp16: bool | None = None,
    max_rows: int | None = None,
    normalize_text: bool | None = None,
    report_to_mlflow: bool = False,
    run_name: str | None = None,
    settings: Settings | None = None,
) -> dict[str, float]:
    cfg = settings or get_settings()
    hp = cfg.training.mlm

    raw_csv = raw_csv if raw_csv is not None else cfg.paths.raw_csv
    out_dir = out_dir if out_dir is not None else cfg.paths.mlm_dir
    base_model = base_model if base_model is not None else hp.base_model
    epochs = epochs if epochs is not None else hp.epochs
    batch_size = batch_size if batch_size is not None else hp.batch_size
    learning_rate = learning_rate if learning_rate is not None else hp.learning_rate
    weight_decay = weight_decay if weight_decay is not None else hp.weight_decay
    max_length = max_length if max_length is not None else hp.max_length
    use_fp16 = use_fp16 if use_fp16 is not None else hp.use_fp16
    max_rows = max_rows if max_rows is not None else hp.max_rows
    normalize_text = (
        normalize_text if normalize_text is not None else hp.normalize_text
    )

    texts = load_corpus(
        raw_csv,
        cfg.data.text_column,
        max_rows=max_rows,
        normalize_text=normalize_text,
    )
    logger.info(
        "pretraining on %d rows from %s (normalized=%s)",
        len(texts),
        raw_csv,
        normalize_text,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    seed = cfg.project["seed"]
    holdout = cfg.data.val_fraction + cfg.data.test_fraction
    split = ds.train_test_split(test_size=holdout, seed=seed)
    rest = split["test"].train_test_split(
        test_size=cfg.data.test_fraction / holdout, seed=seed
    )
    train_ds, val_ds, test_ds = split["train"], rest["train"], rest["test"]
    logger.info(
        "split: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds)
    )

    model = AutoModelForMaskedLM.from_pretrained(base_model)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=hp.mlm_probability
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=use_fp16,
        save_total_limit=2,
        run_name=run_name,
        report_to="mlflow" if report_to_mlflow else "none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    results = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if "test_loss" in results:
        try:
            results["test_perplexity"] = math.exp(results["test_loss"])
        except OverflowError:
            results["test_perplexity"] = float("inf")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Continued MLM pretraining.")
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--no-normalize",
        dest="normalize_text",
        action="store_false",
        default=None,
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--track",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    metrics = train_mlm(
        args.raw_csv,
        args.out_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        max_rows=args.max_rows,
        normalize_text=args.normalize_text,
        report_to_mlflow=args.track,
        run_name=args.run_name,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
