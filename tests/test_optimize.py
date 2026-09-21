from pathlib import Path

import pytest

from asap.config import get_settings
from asap.optimize.export_onnx import DYNAMIC_AXES, export_onnx


def test_onnx_dir_is_configured():
    assert get_settings().paths.onnx_dir == Path("models/onnx")


def test_export_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="no classifier checkpoint"):
        export_onnx(src_dir=tmp_path / "nope", out_dir=tmp_path / "out")


def test_both_inputs_declare_a_dynamic_sequence_axis():
    """Hand-rolled axes are this export's main risk. A fixed sequence axis
    would break every batch not padded to exactly the traced length."""
    for name in ("input_ids", "attention_mask"):
        assert DYNAMIC_AXES[name][0] == "batch"
        assert DYNAMIC_AXES[name][1] == "sequence"

    assert DYNAMIC_AXES["logits"] == {0: "batch"}


@pytest.mark.slow
def test_export_writes_a_graph_a_tokenizer_and_a_config(tmp_path):
    cfg = get_settings()
    if not cfg.paths.cls_dir.is_dir():
        pytest.skip(f"no classifier checkpoint at {cfg.paths.cls_dir}")

    out = tmp_path / "onnx"
    graph = export_onnx(src_dir=cfg.paths.cls_dir, out_dir=out)

    assert graph.is_file()
    assert graph.name == "model.onnx"
    assert (out / "config.json").is_file()
    assert (out / "tokenizer.json").is_file()


@pytest.mark.slow
def test_exported_graph_declares_dynamic_dimensions(tmp_path):
    """Reads the written graph rather than trusting the export call, so a
    silently-static axis cannot pass."""
    cfg = get_settings()
    if not cfg.paths.cls_dir.is_dir():
        pytest.skip(f"no classifier checkpoint at {cfg.paths.cls_dir}")

    # importorskip, not a bare import: the onnx package comes from the export
    # extra, which CI does not install. A bare import here raised
    # ModuleNotFoundError instead of skipping, failing the run.
    onnx = pytest.importorskip("onnx", reason="needs the export extra")

    graph = export_onnx(src_dir=cfg.paths.cls_dir, out_dir=tmp_path / "onnx")
    model = onnx.load(str(graph))

    declared = {
        i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in model.graph.input
    }

    assert set(declared) == {"input_ids", "attention_mask"}
    for name in declared:
        assert declared[name] == ["batch", "sequence"], declared[name]
