from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from asap.config import Settings, get_settings


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted")),
    }


def _load(data_dir: Path, split: str) -> Dataset:
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def train(
    data_dir: Path | None = None,
    base_model: str | None = None,
    out_dir: Path | None = None,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    max_length: int | None = None,
    use_fp16: bool | None = None,
    report_to_mlflow: bool = False,
    run_name: str | None = None,
    settings: Settings | None = None,
) -> dict[str, float]:
    cfg = settings or get_settings()
    hp = cfg.training.classifier

    data_dir = data_dir if data_dir is not None else cfg.paths.processed_dir
    base_model = base_model if base_model is not None else hp.base_model
    out_dir = out_dir if out_dir is not None else cfg.paths.cls_dir
    epochs = epochs if epochs is not None else hp.epochs
    batch_size = batch_size if batch_size is not None else hp.batch_size
    learning_rate = learning_rate if learning_rate is not None else hp.learning_rate
    weight_decay = weight_decay if weight_decay is not None else hp.weight_decay
    max_length = max_length if max_length is not None else hp.max_length
    use_fp16 = use_fp16 if use_fp16 is not None else hp.use_fp16

    id2label = dict(cfg.data.labels)
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    splits = {s: _load(data_dir, s).map(tokenize, batched=True) for s in ("train", "val", "test")}

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=hp.num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model=hp.metric_for_best_model,
        greater_is_better=hp.greater_is_better,
        save_total_limit=hp.save_total_limit,
        run_name=run_name,
        report_to="mlflow" if report_to_mlflow else "none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp.early_stopping_patience)],
    )

    trainer.train()
    test_metrics = trainer.evaluate(splits["test"], metric_key_prefix="test")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--track",
        action="store_true",
        help="Log the run to MLflow. Requires mlflow to be installed (M2).",
    )
    args = ap.parse_args()

    metrics = train(
        args.data_dir,
        args.base_model,
        args.out_dir,
        epochs=args.epochs,
        run_name=args.run_name,
        report_to_mlflow=args.track,
    )
    print(json.dumps(metrics, indent=2))