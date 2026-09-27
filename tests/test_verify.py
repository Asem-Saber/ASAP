import os

import pytest

from asap.config import get_settings

pytestmark = pytest.mark.slow

SAMPLE = ["سيء", "المنتج رائع جدا وانصح به بشدة", "خدمة " * 60]

SOURCE_ENV = "ASAP_VERIFY_SOURCE"


def _served_graph(cfg):
    graph = cfg.paths.onnx_dir / "model.onnx"
    if not graph.is_file():
        pytest.skip(f"no exported graph at {cfg.paths.onnx_dir}; run asap-export-onnx")
    return cfg.paths.onnx_dir


def test_batching_does_not_change_predictions():
    """The dynamic-axes gate, and the one that needs no source checkpoint."""
    pytest.importorskip("onnxruntime")
    from asap.optimize.verify import padding_consistency

    cfg = get_settings()
    onnx_dir = _served_graph(cfg)

    delta = padding_consistency(onnx_dir, SAMPLE, max_length=cfg.inference.max_length)

    assert delta == 0.0, f"batching shifted logits by {delta:.3e}"


def test_exported_graph_matches_its_source_checkpoint():
    """Full parity against the torch checkpoint the graph came from."""
    pytest.importorskip("onnxruntime")
    from pathlib import Path

    from asap.optimize.verify import logits_match

    cfg = get_settings()
    onnx_dir = _served_graph(cfg)

    configured = os.environ.get(SOURCE_ENV)
    torch_dir = Path(configured) if configured else cfg.paths.cls_dir
    if not (torch_dir / "config.json").is_file():
        pytest.skip(
            f"source checkpoint {torch_dir} not found, so torch parity is "
            f"unchecked. Set {SOURCE_ENV} to the directory the graph in "
            f"{onnx_dir} was exported from."
        )

    result = logits_match(
        torch_dir, onnx_dir, SAMPLE, max_length=cfg.inference.max_length
    )

    assert result["label_agreement"] == 1.0, result
    assert result["max_abs_delta"] < 1e-4, result
    assert result["padding_delta"] == 0.0, result


def test_verify_helpers_reject_an_empty_text_list():
    pytest.importorskip("onnxruntime")
    from asap.optimize.verify import logits_match, padding_consistency

    cfg = get_settings()
    with pytest.raises(ValueError, match="at least one text"):
        logits_match(cfg.paths.cls_dir, cfg.paths.onnx_dir, [], max_length=128)
    with pytest.raises(ValueError, match="at least one text"):
        padding_consistency(cfg.paths.onnx_dir, [], max_length=128)
