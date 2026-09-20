from types import SimpleNamespace

import numpy as np
import pytest

from asap.config import get_settings
from asap.inference import load_model, predictor, reset_model
from asap.inference.predictor import OnnxPredictor, _softmax


@pytest.fixture
def stub_session(tmp_path, monkeypatch):
    """Patch AutoTokenizer and ort.InferenceSession with fakes and return the
    shared call-recording dict. After this, OnnxPredictor(model_dir=tmp_path)
    constructs without a real graph."""
    (tmp_path / "model.onnx").write_bytes(b"not-a-real-graph")
    calls: dict = {"runs": []}

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            calls["texts"] = texts
            calls["kwargs"] = kwargs
            n = len(texts)
            return {
                "input_ids": np.ones((n, 4), dtype=np.int64),
                "attention_mask": np.ones((n, 4), dtype=np.int64),
                "token_type_ids": np.zeros((n, 4), dtype=np.int64),
            }

    class FakeSession:
        def __init__(self, path, options=None, providers=None):
            calls["session_path"] = path
            calls["options"] = options
            calls["providers"] = providers

        def get_providers(self):
            return list(calls["providers"])

        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, _outputs, feed):
            calls["feed"] = feed
            n = feed["input_ids"].shape[0]
            calls["runs"].append(n)
            return [np.tile(np.array([[2.0, 4.0]]), (n, 1))]

    monkeypatch.setattr(
        predictor,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
    )
    monkeypatch.setattr(predictor.ort, "InferenceSession", FakeSession)

    return calls


@pytest.fixture
def stub_predictor(tmp_path, stub_session):
    """An OnnxPredictor over the fake session, so label mapping, normalization,
    feed construction and chunking are testable without a real graph."""
    p = OnnxPredictor(model_dir=tmp_path)
    p.calls = stub_session
    return p


def test_softmax_rows_sum_to_one():
    probs = _softmax(np.array([[2.0, 4.0], [10.0, -10.0]]))
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert probs[0, 1] == pytest.approx(0.880797, abs=1e-5)


def test_softmax_is_stable_on_large_logits():
    probs = _softmax(np.array([[1000.0, 999.0]]))
    assert np.isfinite(probs).all()


def test_missing_graph_raises_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match="no ONNX graph"):
        OnnxPredictor(model_dir=tmp_path / "empty")


def test_predict_maps_argmax_through_config_labels(stub_predictor):
    result = stub_predictor.predict("المنتج رائع")
    assert result["sentiment"] == get_settings().data.labels[1]
    assert result["confidence"] == pytest.approx(0.880797, abs=1e-5)


def test_predict_normalizes_before_tokenizing(stub_predictor):
    stub_predictor.predict("المنتج رااااائع")
    assert stub_predictor.calls["texts"] == ["المنتج رائع"]


def test_feed_omits_inputs_the_graph_does_not_declare(stub_predictor):
    stub_predictor.predict("المنتج رائع")
    assert set(stub_predictor.calls["feed"]) == {"input_ids", "attention_mask"}


def test_feed_is_int64(stub_predictor):
    stub_predictor.predict("المنتج رائع")
    assert stub_predictor.calls["feed"]["input_ids"].dtype == np.int64


def test_truncation_settings_reach_the_tokenizer(stub_predictor):
    cfg = get_settings()
    stub_predictor.predict("المنتج رائع")
    assert stub_predictor.calls["kwargs"]["max_length"] == cfg.inference.max_length
    assert stub_predictor.calls["kwargs"]["truncation"] == cfg.inference.truncate
    assert stub_predictor.calls["kwargs"]["return_tensors"] == "np"


def test_predict_batch_preserves_order_and_length(stub_predictor):
    results = stub_predictor.predict_batch(["نص اول", "نص ثاني", "نص ثالث"])
    assert len(results) == 3
    assert all(r["sentiment"] == get_settings().data.labels[1] for r in results)


def test_predict_batch_on_empty_list_never_touches_the_session(stub_predictor):
    assert stub_predictor.predict_batch([]) == []
    assert stub_predictor.calls["runs"] == []


def test_predict_batch_chunks_by_configured_batch_size(stub_predictor):
    stub_predictor.batch_size = 2
    stub_predictor.predict_batch(["a", "b", "c", "d", "e"])
    assert stub_predictor.calls["runs"] == [2, 2, 1]


def test_configured_thread_count_reaches_the_session_options(stub_predictor):
    """Unset, ORT spawns one thread per physical core, which is 4.4x slower
    than 4 threads for this model. The setting has to actually arrive."""
    expected = get_settings().inference.intra_op_num_threads
    assert stub_predictor.intra_op_num_threads == expected
    assert stub_predictor.calls["options"].intra_op_num_threads == expected


def test_thread_count_can_be_overridden_per_instance(tmp_path, stub_session):
    p = OnnxPredictor(model_dir=tmp_path, intra_op_num_threads=2)
    assert p.intra_op_num_threads == 2
    assert stub_session["options"].intra_op_num_threads == 2


def test_zero_threads_leaves_the_ort_default_untouched(tmp_path, stub_session):
    """0 means 'let ORT decide', so we must not write anything into options."""
    p = OnnxPredictor(model_dir=tmp_path, intra_op_num_threads=0)
    assert p.intra_op_num_threads == 0
    assert stub_session["options"].intra_op_num_threads == 0


def test_providers_attribute_is_exposed_for_health(stub_predictor):
    assert stub_predictor.providers == ["CPUExecutionProvider"]


def test_provider_is_wrapped_in_a_list_for_ort(stub_predictor):
    """ORT's contract is a sequence. It happens to tolerate a bare string, but
    that is not something to depend on."""
    assert stub_predictor.provider == get_settings().inference.provider
    assert stub_predictor.calls["providers"] == [get_settings().inference.provider]


def test_provider_can_be_overridden_per_instance(tmp_path, stub_session):
    OnnxPredictor(model_dir=tmp_path, provider="CUDAExecutionProvider")
    assert stub_session["providers"] == ["CUDAExecutionProvider"]


def test_unknown_provider_name_is_rejected_by_config():
    """ONNX Runtime silently accepts a bogus provider and falls back to CPU, so
    a typo would degrade serving with no error. The Literal has to catch it."""
    from pydantic import ValidationError

    from asap.config import InferenceConfig

    with pytest.raises(ValidationError):
        InferenceConfig(
            model_dir="models/onnx",
            model_file="model.onnx",
            provider="CPUExecutionProvder",
            max_length=128,
            batch_size=32,
            truncate=True,
        )


def test_warmup_runs_one_inference(stub_predictor):
    stub_predictor.warmup()
    assert stub_predictor.calls["runs"] == [1]


def test_load_model_is_cached(monkeypatch):
    built = []

    class FakeModel:
        def __init__(self, settings=None):
            built.append(settings)

        def warmup(self):
            pass

    monkeypatch.setattr(predictor, "OnnxPredictor", FakeModel)
    reset_model()
    try:
        assert load_model() is load_model()
        assert len(built) == 1
    finally:
        reset_model()


@pytest.mark.slow
def test_real_graph_classifies_arabic_sentiment():
    cfg = get_settings()
    graph = cfg.inference.model_dir / cfg.inference.model_file
    if not graph.is_file():
        pytest.skip(f"no exported graph at {graph}; run asap-export-onnx")

    reset_model()
    try:
        model = load_model()
        assert model.predict("المنتج رائع")["sentiment"] == "positive"
        assert model.predict("خدمة سيئة جدا")["sentiment"] == "negative"

        batch = model.predict_batch(["المنتج رائع", "خدمة سيئة جدا"])
        assert [r["sentiment"] for r in batch] == ["positive", "negative"]
        assert all(0.0 <= r["confidence"] <= 1.0 for r in batch)
    finally:
        reset_model()
