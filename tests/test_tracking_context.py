import hashlib
import subprocess
import sys
from pathlib import Path

from asap.config import get_settings
from asap.tracking.context import (
    RunContext,
    build_run_context,
    read_data_hash,
    sha256_file,
)


def test_tracking_config_is_typed():
    t = get_settings().tracking
    assert t.experiment == "asap-sentiment"
    assert t.registered_model == "asap-sentiment-clf"
    assert t.primary_metric == "test_f1"
    assert t.greater_is_better is True
    assert isinstance(t.artifact_dir, Path)


def test_experiments_dir_is_separate_from_the_served_checkpoint():
    """A variant run must never overwrite what the API serves."""
    paths = get_settings().paths
    assert paths.experiments_dir == Path("models/experiments")
    assert paths.experiments_dir != paths.cls_dir


def test_sha256_file_matches_hashlib(tmp_path):
    blob = b"arabic-sentiment" * 5000
    target = tmp_path / "model.safetensors"
    target.write_bytes(blob)

    assert sha256_file(target) == hashlib.sha256(blob).hexdigest()


def test_sha256_file_is_correct_across_multiple_chunks(tmp_path):
    """Exercises the chunked read loop rather than a single-chunk file, since
    checkpoints here are ~500 MB and an off-by-one in the loop would only show
    up past the first chunk."""
    blob = bytes(range(256)) * 20_000  # ~5 MB, several 1 MiB chunks
    target = tmp_path / "big.bin"
    target.write_bytes(blob)

    assert sha256_file(target) == hashlib.sha256(blob).hexdigest()


def test_sha256_file_handles_an_empty_file(tmp_path):
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")

    assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


def test_read_data_hash_returns_none_when_lock_is_absent(tmp_path):
    assert read_data_hash(tmp_path / "dvc.lock") is None


def test_read_data_hash_returns_none_when_stage_is_missing(tmp_path):
    lock = tmp_path / "dvc.lock"
    lock.write_text("schema: '2.0'\nstages:\n  train:\n    cmd: echo hi\n")

    assert read_data_hash(lock) is None


def test_read_data_hash_reads_the_prepare_output(tmp_path):
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\n"
        "stages:\n"
        "  prepare:\n"
        "    cmd: python -m asap.data.build\n"
        "    outs:\n"
        "    - path: data/processed\n"
        "      md5: abc123.dir\n"
    )

    assert read_data_hash(lock) == "abc123.dir"


def test_read_data_hash_ignores_outputs_other_than_the_split_dir(tmp_path):
    """The prepare stage may gain other outputs; only data/processed is the
    corpus the runs are keyed on."""
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "stages:\n"
        "  prepare:\n"
        "    outs:\n"
        "    - path: reports/prepare.json\n"
        "      md5: deadbeef\n"
        "    - path: data/processed\n"
        "      md5: abc123.dir\n"
    )

    assert read_data_hash(lock) == "abc123.dir"


def test_read_data_hash_survives_malformed_yaml(tmp_path):
    """A broken lock file must not take down a training run."""
    lock = tmp_path / "dvc.lock"
    lock.write_text("stages: [this is not: valid: yaml")

    assert read_data_hash(lock) is None


def test_build_run_context_records_the_variant_and_params():
    ctx = build_run_context("baseline", {"learning_rate": 2e-5})

    assert isinstance(ctx, RunContext)
    assert ctx.variant == "baseline"
    assert ctx.params["learning_rate"] == 2e-5
    assert len(ctx.git_sha) == 40
    assert isinstance(ctx.git_dirty, bool)


def test_build_run_context_copies_params(tmp_path):
    """The context is frozen; a caller mutating its dict afterwards must not
    retroactively change what the run recorded."""
    params = {"epochs": 5}
    ctx = build_run_context("baseline", params)
    params["epochs"] = 99

    assert ctx.params["epochs"] == 5


def test_build_run_context_omits_the_data_hash_when_dvc_is_absent(monkeypatch, tmp_path):
    """Phase 1 has no dvc.lock. The tag must be absent, never 'unknown' — a
    placeholder would match queries looking for a real hash."""
    monkeypatch.chdir(tmp_path)

    ctx = build_run_context("baseline", {}, settings=get_settings())

    assert ctx.data_hash is None


def test_context_module_does_not_import_mlflow():
    """The whole point of this module's purity: its tests run in CI, where
    mlflow is not installed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asap.tracking.context, sys; "
            "sys.exit(1 if 'mlflow' in sys.modules else 0)",
        ],
        capture_output=True,
    )

    assert result.returncode == 0, "asap.tracking.context imported mlflow"
