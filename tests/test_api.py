import pytest
from fastapi.testclient import TestClient

from asap.api import deps
from asap.api.app import create_app
from asap.config import get_settings
from asap.inference import predictor

API = get_settings().api


class FakeModel:
    device = "cpu"
    dtype = "float32"

    def predict(self, text: str):
        return {"sentiment": "positive", "confidence": 0.99}

    def predict_batch(self, texts: list[str]):
        return [{"sentiment": "negative", "confidence": 0.98} for _ in texts]


@pytest.fixture
def client():
    """No lifespan: the app is exercised with a fake model rather than the real
    checkpoint. See test_real_model_endpoint for the end-to-end path."""
    app = create_app()
    app.dependency_overrides[deps.get_model] = FakeModel
    return TestClient(app)


def test_predict_returns_sentiment_and_confidence(client):
    response = client.post("/predict", json={"text": "المنتج رائع"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "positive", "confidence": 0.99}


def test_predict_batch_returns_one_prediction_per_input(client):
    response = client.post("/predict/batch", json={"texts": ["اول", "ثاني", "ثالث"]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_oversized_text(client):
    response = client.post("/predict", json={"text": "ا" * (API.max_text_chars + 1)})
    assert response.status_code == 422


def test_predict_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_predict_batch_rejects_oversized_batch(client):
    response = client.post(
        "/predict/batch", json={"texts": ["نص"] * (API.max_batch_items + 1)}
    )
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(predictor, "sentiment_model", None)
    client = TestClient(create_app())

    response = client.post("/predict", json={"text": "المنتج رائع"})
    assert response.status_code == 503


def test_health_reports_loading_before_startup(monkeypatch):
    monkeypatch.setattr(predictor, "sentiment_model", None)
    client = TestClient(create_app())

    body = client.get("/").json()
    assert body["status"] == "loading"
    assert body["model_loaded"] is False
    assert body["model_dir"] == str(get_settings().inference.model_dir)
    assert body["dtype"] == get_settings().inference.dtype


def test_cors_middleware_added_only_when_origins_configured(monkeypatch):
    cfg = get_settings()
    monkeypatch.setattr(cfg.api, "cors_origins", ["https://example.com"])
    client = TestClient(create_app())

    response = client.get("/", headers={"Origin": "https://example.com"})
    assert response.headers["access-control-allow-origin"] == "https://example.com"


def test_no_cors_headers_by_default(client):
    response = client.get("/", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.gpu
@pytest.mark.slow
def test_real_model_endpoint():
    cfg = get_settings()
    if not cfg.inference.model_dir.exists():
        pytest.skip(f"no checkpoint at {cfg.inference.model_dir}")

    predictor.reset_model()
    try:
        # The context manager runs the lifespan, which loads the real model.
        with TestClient(create_app()) as client:
            assert client.get("/").json()["model_loaded"] is True

            assert (
                client.post("/predict", json={"text": "المنتج رائع"}).json()["sentiment"]
                == "positive"
            )

            batch = client.post(
                "/predict/batch", json={"texts": ["المنتج رائع", "خدمة سيئة جدا"]}
            ).json()["predictions"]
            assert [p["sentiment"] for p in batch] == ["positive", "negative"]
    finally:
        predictor.reset_model()
