import pytest

from asap.config import get_settings
from asap.optimize.quantize import (
    MAX_F1_DROP,
    MIN_LABEL_AGREEMENT,
    QUANTIZED_FILENAME,
    quantize_onnx,
)


def test_quantized_graph_sits_beside_the_fp32_graph():
    """Serving picks a graph through inference.model_file, so both must live in
    the same directory for the switch to be a config change only."""
    assert QUANTIZED_FILENAME == "model_quantized.onnx"


def test_quantizing_never_overwrites_the_served_graph():
    """A quantized graph written over model.onnx would silently replace the
    fp32 baseline every comparison is measured against."""
    assert QUANTIZED_FILENAME != get_settings().inference.model_file


def test_gate_thresholds_match_the_serving_design():
    assert MIN_LABEL_AGREEMENT == 0.99
    assert MAX_F1_DROP == 0.005


def test_quantize_rejects_a_missing_graph(tmp_path):
    with pytest.raises(FileNotFoundError, match="no ONNX graph"):
        quantize_onnx(src=tmp_path / "absent.onnx", out=tmp_path / "out.onnx")


def test_quantize_defaults_to_the_configured_serving_directory(tmp_path, monkeypatch):
    """The default source is whatever serving currently loads; the default
    output is its INT8 sibling."""
    cfg = get_settings()
    monkeypatch.setattr(cfg.inference, "model_dir", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        quantize_onnx(settings=cfg)

    assert str(tmp_path / cfg.inference.model_file) in str(excinfo.value)


@pytest.mark.slow
def test_quantization_shrinks_the_graph(tmp_path):
    cfg = get_settings()
    src = cfg.inference.model_dir / cfg.inference.model_file
    if not src.is_file():
        pytest.skip(f"no exported graph at {src}; run asap-export-onnx")

    out = quantize_onnx(src=src, out=tmp_path / QUANTIZED_FILENAME)

    assert out.is_file()
    assert out.stat().st_size < src.stat().st_size


@pytest.mark.slow
def test_graphs_match_compares_the_two_graphs(tmp_path):
    """Checks the comparison works, not that this model passes the gate.

    Whether a given graph clears MIN_LABEL_AGREEMENT is a property of the model,
    not of the code — that verdict belongs to `asap-quantize --check`, which
    exits non-zero on failure. Asserting it here would turn a quality regression
    into a unit-test failure.
    """
    import json

    from asap.optimize.verify import graphs_match

    cfg = get_settings()
    src = cfg.inference.model_dir / cfg.inference.model_file
    fixture = cfg.paths.bench_sample
    if not src.is_file():
        pytest.skip(f"no exported graph at {src}; run asap-export-onnx")
    if not fixture.is_file():
        pytest.skip(f"no sample at {fixture}; run scripts/sample_data.py")

    quantize_onnx(src=src, out=cfg.inference.model_dir / QUANTIZED_FILENAME)

    with fixture.open(encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle][:16]

    result = graphs_match(
        cfg.inference.model_dir,
        texts,
        max_length=cfg.inference.max_length,
        candidate_file=QUANTIZED_FILENAME,
    )

    assert 0.0 <= result["label_agreement"] <= 1.0
    assert result["max_abs_delta"] >= 0.0


def test_graphs_match_rejects_an_empty_text_list():
    from asap.optimize.verify import graphs_match

    with pytest.raises(ValueError, match="at least one text"):
        graphs_match(get_settings().inference.model_dir, [], max_length=128)
