import argparse
import json
import logging

import mlflow
import pandas as pd

from asap.config import get_settings
from asap.optimize.export_onnx import export_onnx
from asap.optimize.verify import logits_match
from asap.tracking import build_run_context
from asap.tracking.mlflow_run import mlflow_run
from asap.training.experiment import load_variants, variant_out_dir
from asap.training.trainer import train

logger = logging.getLogger(__name__)

PARITY_SAMPLE = 64

_BOOKKEEPING = ("_runtime", "_samples_per_second", "_steps_per_second")


def quality_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        k: v
        for k, v in metrics.items()
        if k != "epoch" and not k.endswith(_BOOKKEEPING)
    }


def parity_texts(cfg, n: int = PARITY_SAMPLE) -> list[str]:
    path = cfg.paths.processed_dir / "test.parquet"
    return pd.read_parquet(path)["text"].head(n).tolist()


def train_variant(name: str, cfg) -> str:
    variant = load_variants()[name]
    out_dir = variant_out_dir(name, settings=cfg)
    params = {
        "base_model": variant.base_model,
        "learning_rate": variant.learning_rate,
        "epochs": variant.epochs,
    }

    with mlflow_run(build_run_context(name, params), settings=cfg):
        metrics = train(
            base_model=variant.base_model,
            out_dir=out_dir,
            epochs=variant.epochs,
            learning_rate=variant.learning_rate,
            report_to_mlflow=True,
            run_name=name,
            settings=cfg,
        )
        mlflow.log_metrics(metrics)

        logged = mlflow.create_external_model(name=name, model_type="transformers")
        mlflow.set_logged_model_tags(
            logged.model_id,
            {"variant": name, "base_model": variant.base_model, "path": str(out_dir)},
        )
        for key, value in quality_metrics(metrics).items():
            mlflow.log_metric(key, value, model_id=logged.model_id)

        logger.info(
            "%s: %s=%.4f",
            name,
            cfg.tracking.primary_metric,
            metrics[cfg.tracking.primary_metric],
        )
        return logged.model_id


def select_best_model(cfg):
    experiment = mlflow.get_experiment_by_name(cfg.tracking.experiment)
    if experiment is None:
        raise SystemExit(f"no experiment named {cfg.tracking.experiment!r}")

    found = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            {
                "field_name": f"metrics.{cfg.tracking.primary_metric}",
                "ascending": not cfg.tracking.greater_is_better,
            }
        ],
        max_results=1,
        output_format="list",
    )
    if not found:
        raise SystemExit("no logged models to choose a best from")
    return found[0]


def export_best_model(model, cfg, texts: list[str]) -> None:
    name = model.tags["variant"]
    out_dir = variant_out_dir(name, settings=cfg)

    graph = export_onnx(out_dir, cfg.paths.onnx_dir, settings=cfg)
    parity = logits_match(
        out_dir, cfg.paths.onnx_dir, texts, max_length=cfg.inference.max_length
    )
    logger.info("export parity for %s: %s", name, parity)

    if parity["label_agreement"] < 1.0 or parity["padding_delta"] > 0:
        raise SystemExit(
            f"the exported graph does not match {name}'s checkpoint ({parity}). "
            "Not publishing it. The export's dynamic axes were only ever "
            "verified against DistilBERT; a 12-layer BERT may trace differently."
        )

    with mlflow.start_run(run_id=model.source_run_id):
        mlflow.log_artifacts(str(cfg.paths.onnx_dir), artifact_path="onnx")
        mlflow.log_metrics(
            {
                "onnx_size_mb": graph.stat().st_size / 1e6,
                "onnx_max_abs_delta": parity["max_abs_delta"],
                "onnx_label_agreement": parity["label_agreement"],
            }
        )
    mlflow.set_logged_model_tags(model.model_id, {"best": "true"})

    version = mlflow.register_model(
        f"models:/{model.model_id}", cfg.tracking.registered_model
    )
    mlflow.MlflowClient().set_registered_model_alias(
        cfg.tracking.registered_model, "best", version.version
    )
    logger.info(
        "published %s as %s v%s; %s now serves it",
        name,
        cfg.tracking.registered_model,
        version.version,
        cfg.paths.onnx_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="subset of variant names")
    parser.add_argument(
        "--no-export",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = get_settings()

    variants = load_variants()
    unknown = set(args.only or []) - set(variants)
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}; have {sorted(variants)}")

    names = [n for n in sorted(variants) if not args.only or n in args.only]
    for name in names:
        logger.info("=== %s ===", name)
        train_variant(name, cfg)

    best_model = select_best_model(cfg)
    scores = {m.key: m.value for m in best_model.metrics}
    print(
        json.dumps({"best_model": best_model.tags["variant"], "metrics": scores}, indent=2)
    )

    if args.no_export:
        logger.info("skipping export; %s stays as it was", cfg.paths.onnx_dir)
        return

    export_best_model(best_model, cfg, parity_texts(cfg))


if __name__ == "__main__":
    main()
