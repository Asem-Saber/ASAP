import json
import re

import httpx
import pytest

from asap.config import get_settings
from asap.ui.client import ApiClient, ApiError

BASE_URL = "http://testserver:8000"


def _client(handler, timeout: float = 5.0) -> ApiClient:
    """An ApiClient whose requests are answered by `handler`, not by a server."""
    return ApiClient(
        base_url=BASE_URL, timeout=timeout, transport=httpx.MockTransport(handler)
    )


def _responds(status_code: int, payload=None, text: str = ""):
    def handler(request: httpx.Request) -> httpx.Response:
        if payload is not None:
            return httpx.Response(status_code, json=payload)
        return httpx.Response(status_code, text=text)

    return handler


def test_predict_parses_a_successful_response():
    client = _client(_responds(200, {"sentiment": "positive", "confidence": 0.99}))

    result = client.predict("المنتج رائع")

    assert result.sentiment == "positive"
    assert result.confidence == pytest.approx(0.99)


def test_predict_posts_the_text_to_the_predict_route():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"sentiment": "negative", "confidence": 0.5})

    _client(handler).predict("خدمة سيئة")

    assert seen["url"] == f"{BASE_URL}/predict"
    assert seen["body"] == {"text": "خدمة سيئة"}


def test_503_explains_that_the_model_is_still_loading():
    """The service warms up before binding the port; a generic error here
    makes a cold demo look broken."""
    client = _client(_responds(503, {"detail": "model is not loaded"}))

    with pytest.raises(ApiError, match=r"still loading.*30-40 seconds"):
        client.predict("المنتج رائع")


def test_422_surfaces_the_validation_detail():
    body = {"detail": [{"msg": "String should have at most 4000 characters"}]}
    client = _client(_responds(422, body))

    with pytest.raises(ApiError, match="at most 4000 characters"):
        client.predict("ا" * 5000)


def test_500_reports_the_status_code():
    client = _client(_responds(500, text="boom"))

    with pytest.raises(ApiError, match="HTTP 500"):
        client.predict("المنتج رائع")


def test_connect_error_names_the_base_url():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ApiError, match=re.escape(BASE_URL) + r".*uv run asap-api"):
        _client(handler).predict("المنتج رائع")


def test_malformed_base_url_raises_api_error_naming_the_setting():
    with pytest.raises(ApiError, match=r"http://host:port.*ASAP_UI__API_URL"):
        ApiClient(base_url="http://host:port", timeout=5.0)


def test_422_with_a_non_json_body_surfaces_the_raw_text():
    client = _client(_responds(422, text="plain failure text"))

    with pytest.raises(ApiError, match="plain failure text"):
        client.predict("المنتج رائع")


def test_422_with_a_string_detail_surfaces_it():
    client = _client(_responds(422, {"detail": "text must not be blank"}))

    with pytest.raises(ApiError, match="text must not be blank"):
        client.predict("المنتج رائع")


def test_422_with_an_unexpected_json_shape_falls_back_to_the_body():
    client = _client(_responds(422, {"errors": ["odd-shape-marker"]}))

    with pytest.raises(ApiError, match="odd-shape-marker"):
        client.predict("المنتج رائع")


def test_bodyless_error_has_no_dangling_colon():
    client = _client(_responds(502))

    with pytest.raises(ApiError, match="502") as excinfo:
        client.predict("المنتج رائع")

    message = str(excinfo.value)
    assert message == message.rstrip()
    assert not message.endswith(":")


def test_redirects_are_followed():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/predict":
            return httpx.Response(308, headers={"location": f"{BASE_URL}/v2/predict"})
        return httpx.Response(200, json={"sentiment": "positive", "confidence": 0.8})

    result = _client(handler).predict("المنتج رائع")

    assert result.sentiment == "positive"


def test_302_redirect_is_refused_rather_than_parsed_as_a_prediction():
    """httpx re-sends a 302 as a bodiless GET; its answer is for no text."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/predict":
            return httpx.Response(302, headers={"location": f"{BASE_URL}/v2/predict"})
        return httpx.Response(200, json={"sentiment": "positive", "confidence": 0.8})

    with pytest.raises(ApiError, match="redirect"):
        _client(handler).predict("المنتج رائع")


def test_307_redirect_is_followed_with_the_original_post_body():
    seen: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content)))
        if request.url.path == "/predict":
            return httpx.Response(307, headers={"location": f"{BASE_URL}/v2/predict"})
        return httpx.Response(200, json={"sentiment": "positive", "confidence": 0.8})

    result = _client(handler).predict("المنتج رائع")

    assert result.sentiment == "positive"
    assert seen[-1] == ("POST", "/v2/predict", {"text": "المنتج رائع"})


def test_422_with_a_list_detail_of_non_dicts_falls_back_to_the_body():
    client = _client(_responds(422, {"detail": ["x-list-marker"]}))

    with pytest.raises(ApiError, match="x-list-marker"):
        client.predict("المنتج رائع")


def test_422_with_a_list_detail_of_non_containers_does_not_raise_type_error():
    client = _client(_responds(422, {"detail": [1]}))

    with pytest.raises(ApiError, match=r"\[1\]"):
        client.predict("المنتج رائع")


def test_timeout_mentions_the_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ApiError, match="timed out after 5s"):
        _client(handler, timeout=5.0).predict("المنتج رائع")


def test_other_transport_errors_also_become_api_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("socket closed", request=request)

    with pytest.raises(ApiError):
        _client(handler).predict("المنتج رائع")


def test_unexpected_response_shape_raises_api_error():
    client = _client(_responds(200, {"label": "positive"}))

    with pytest.raises(ApiError, match="does not understand"):
        client.predict("المنتج رائع")


def test_non_json_success_body_raises_api_error():
    client = _client(_responds(200, text="<html>not json</html>"))

    with pytest.raises(ApiError, match="does not understand"):
        client.predict("المنتج رائع")


def test_from_settings_derives_the_url_from_the_api_section(monkeypatch):
    cfg = get_settings()
    monkeypatch.setattr(cfg.ui, "api_url", None)

    client = ApiClient.from_settings(cfg)

    assert client.base_url == f"http://{cfg.api.host}:{cfg.api.port}"
    assert client.timeout == cfg.ui.request_timeout
    client.close()


def test_from_settings_prefers_an_explicit_ui_api_url(monkeypatch):
    cfg = get_settings()
    monkeypatch.setattr(cfg.ui, "api_url", "https://demo.example.com/")

    client = ApiClient.from_settings(cfg)

    assert client.base_url == "https://demo.example.com"
    client.close()
