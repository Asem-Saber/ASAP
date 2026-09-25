import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow

from asap.config import Settings, get_settings
from asap.tracking.context import RunContext

logger = logging.getLogger(__name__)

PARAM_PREFIX = "variant."


@contextmanager
def mlflow_run(ctx: RunContext, *, settings: Settings | None = None) -> Iterator:
    cfg = settings or get_settings()
    t = cfg.tracking

    mlflow.set_tracking_uri(t.uri)

    if mlflow.get_experiment_by_name(t.experiment) is None:
        artifact_dir = Path(t.artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(t.experiment, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(t.experiment)

    with mlflow.start_run(run_name=ctx.variant) as run:
        tags = {
            "variant": ctx.variant,
            "git_sha": ctx.git_sha,
            "git_dirty": str(ctx.git_dirty),
        }
        if ctx.data_hash is not None:
            tags["data_hash"] = ctx.data_hash
        mlflow.set_tags(tags)
        mlflow.log_params({f"{PARAM_PREFIX}{k}": v for k, v in ctx.params.items()})

        yield run
