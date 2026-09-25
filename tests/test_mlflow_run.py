import pytest

from asap.config import get_settings
from asap.tracking.context import RunContext

mlflow = pytest.importorskip("mlflow")

from asap.tracking.mlflow_run import mlflow_run  # noqa: E402


@pytest.fixture
def ctx():
    return RunContext(
        variant="baseline",
        git_sha="a" * 40,
        git_dirty=False,
        data_hash=None,
        params={"learning_rate": 2e-5, "epochs": 5},
    )


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = get_settings()
    monkeypatch.setattr(c.tracking, "uri", f"sqlite:///{tmp_path / 'mlruns.db'}")
    monkeypatch.setattr(c.tracking, "artifact_dir", tmp_path / "artifacts")
    monkeypatch.setattr(c.tracking, "experiment", f"t-{tmp_path.name}")
    return c


def test_context_tags_are_applied(ctx, cfg):
    with mlflow_run(ctx, settings=cfg) as run:
        run_id = run.info.run_id

    tags = mlflow.get_run(run_id).data.tags
    assert tags["variant"] == "baseline"
    assert tags["git_sha"] == "a" * 40
    assert tags["git_dirty"] == "False"


def test_absent_data_hash_is_omitted_not_placeheld(ctx, cfg):
    """'none' as a placeholder is indistinguishable from a real value and
    matches queries looking for runs that have a hash."""
    with mlflow_run(ctx, settings=cfg) as run:
        run_id = run.info.run_id

    assert "data_hash" not in mlflow.get_run(run_id).data.tags


def test_present_data_hash_is_tagged(cfg):
    ctx = RunContext("v", "b" * 40, False, "abc123.dir", {})
    with mlflow_run(ctx, settings=cfg) as run:
        run_id = run.info.run_id

    assert mlflow.get_run(run_id).data.tags["data_hash"] == "abc123.dir"


def test_params_are_namespaced(ctx, cfg):
    """HF's MLflowCallback logs ~100 TrainingArguments entries into the same
    run. Un-prefixed, the ones the matrix declared are lost in it."""
    with mlflow_run(ctx, settings=cfg) as run:
        run_id = run.info.run_id

    params = mlflow.get_run(run_id).data.params
    assert params["variant.epochs"] == "5"
    assert "epochs" not in params


def test_run_name_is_the_variant(ctx, cfg):
    with mlflow_run(ctx, settings=cfg) as run:
        run_id = run.info.run_id

    assert mlflow.get_run(run_id).info.run_name == "baseline"


def test_artifact_location_honours_config(ctx, cfg):
    """set_experiment alone ignores tracking.artifact_dir — the directory has
    to be passed at creation or the setting is silently dead."""
    with mlflow_run(ctx, settings=cfg):
        pass

    exp = mlflow.get_experiment_by_name(cfg.tracking.experiment)
    assert cfg.tracking.artifact_dir.name in exp.artifact_location


def test_existing_experiment_is_reused(ctx, cfg):
    with mlflow_run(ctx, settings=cfg) as first:
        first_exp = first.info.experiment_id
    with mlflow_run(ctx, settings=cfg) as second:
        second_exp = second.info.experiment_id

    assert first_exp == second_exp


def test_run_is_active_inside_the_block(ctx, cfg):
    """HF's MLflowCallback attaches to an already-active run. If ours is not
    active here, per-epoch metrics go nowhere."""
    with mlflow_run(ctx, settings=cfg) as run:
        active = mlflow.active_run()
        assert active is not None
        assert active.info.run_id == run.info.run_id


def test_run_is_closed_on_exit(ctx, cfg):
    with mlflow_run(ctx, settings=cfg) as run:
        run_id = run.info.run_id

    assert mlflow.active_run() is None
    assert mlflow.get_run(run_id).info.status == "FINISHED"


def test_metrics_logged_inside_the_block_land_on_the_run(ctx, cfg):
    """The script logs train() metrics through the ambient run rather than a
    handle, so the ambient run has to be the one we opened."""
    with mlflow_run(ctx, settings=cfg) as run:
        mlflow.log_metrics({"test_f1": 0.912})
        run_id = run.info.run_id

    assert mlflow.get_run(run_id).data.metrics["test_f1"] == pytest.approx(0.912)


def test_exception_marks_the_run_failed(ctx, cfg):
    with pytest.raises(RuntimeError):
        with mlflow_run(ctx, settings=cfg) as run:
            run_id = run.info.run_id
            raise RuntimeError("training blew up")

    assert mlflow.active_run() is None
    assert mlflow.get_run(run_id).info.status in {"FAILED", "KILLED"}
