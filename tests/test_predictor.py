import pytest

from asap.config import get_settings
from asap.inference import load_model, reset_model
from asap.inference.predictor import (
    SentimentModel,
    _resolve_device,
    _resolve_dtype,
)


@pytest.fixture
def stub_model(monkeypatch):
    """A SentimentModel whose classifier is a stub, so the label-mapping and
    normalization logic can be tested without loading a checkpoint."""
    calls: list = []
    pipeline_kwargs: dict = {}

    def fake_pipeline(*args, **kwargs):
        pipeline_kwargs.update(kwargs)

        def classify(inputs):
            calls.append(inputs)
            if isinstance(inputs, str):
                return [{"label": "positive", "score": 0.99}]
            return [{"label": "negative", "score": 0.98} for _ in inputs]

        return classify

    monkeypatch.setattr("asap.inference.predictor.pipeline", fake_pipeline)
    model = SentimentModel()
    model.calls = calls
    model.pipeline_kwargs = pipeline_kwargs
    return model


def test_predict_returns_sentiment_and_confidence(stub_model):
    assert stub_model.predict("المنتج رائع") == {
        "sentiment": "positive",
        "confidence": 0.99,
    }


def test_predict_normalizes_before_classifying(stub_model):
    stub_model.predict("المنتج رااااائع")
    assert stub_model.calls[0] == "المنتج رائع"


def test_predict_maps_label_ids_when_model_has_no_id2label(stub_model):
    assert stub_model._to_label("LABEL_0") == "negative"
    assert stub_model._to_label("LABEL_1") == "positive"


def test_predict_batch_preserves_order(stub_model):
    results = stub_model.predict_batch(["نص اول", "نص ثاني"])
    assert len(results) == 2
    assert all(r["sentiment"] == "negative" for r in results)


def test_predict_batch_on_empty_list(stub_model):
    assert stub_model.predict_batch([]) == []


def test_configured_dtype_is_passed_to_the_pipeline(stub_model):
    assert stub_model.pipeline_kwargs["dtype"] == get_settings().inference.dtype
    assert stub_model.dtype == get_settings().inference.dtype


def test_float16_falls_back_to_float32_on_cpu():
    assert _resolve_dtype("float16", "cpu") == "float32"
    assert _resolve_dtype("float16", "cuda") == "float16"
    assert _resolve_dtype("auto", "cpu") == "auto"
    assert _resolve_dtype("float32", "cpu") == "float32"


def test_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert _resolve_device("cuda") == "cpu"
    assert _resolve_device("cpu") == "cpu"


def test_load_model_is_cached(monkeypatch):
    built = []

    class FakeModel:
        def __init__(self, settings=None):
            built.append(settings)

        def warmup(self):
            pass

    monkeypatch.setattr("asap.inference.predictor.SentimentModel", FakeModel)
    reset_model()
    try:
        assert load_model() is load_model()
        assert len(built) == 1
    finally:
        reset_model()


@pytest.mark.gpu
@pytest.mark.slow
def test_real_checkpoint_classifies_arabic_sentiment():
    cfg = get_settings()
    if not cfg.inference.model_dir.exists():
        pytest.skip(f"no checkpoint at {cfg.inference.model_dir}")

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
