import httpx
from pydantic import ValidationError

from asap.api.schemas import Prediction
from asap.config import Settings, get_settings

_HTTP_OK = 200
_HTTP_UNPROCESSABLE = 422
_HTTP_UNAVAILABLE = 503
_BODY_SNIPPET_CHARS = 200


class ApiError(Exception):
    pass


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:_BODY_SNIPPET_CHARS]

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict) and "msg" in first:
            return str(first["msg"])
    if isinstance(detail, str):
        return detail
    return response.text[:_BODY_SNIPPET_CHARS]


class ApiClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        try:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                transport=transport,
                follow_redirects=True,
            )
        except (httpx.InvalidURL, ValueError) as exc:
            raise ApiError(
                f"The API address {self.base_url!r} is not a valid URL. "
                "Check the ASAP_UI__API_URL setting."
            ) from exc

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ApiClient":
        cfg = settings or get_settings()
        base_url = cfg.ui.api_url or f"http://{cfg.api.host}:{cfg.api.port}"
        return cls(base_url=base_url, timeout=cfg.ui.request_timeout)

    def predict(self, text: str) -> Prediction:
        try:
            response = self._client.post("/predict", json={"text": text})
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the API at {self.base_url}. "
                "Start it with `uv run asap-api`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(
                f"The request timed out after {self.timeout:g}s. "
                "The service may still be warming up."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"The request to {self.base_url} failed: {exc}") from exc

        if response.request.method != "POST":
            raise ApiError(
                f"The API address {self.base_url} redirected the request in a "
                "way this UI will not follow blindly, so the text was not "
                "analysed. Check the ASAP_UI__API_URL setting."
            )

        if response.status_code != _HTTP_OK:
            raise ApiError(self._message_for(response))

        try:
            return Prediction.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise ApiError(
                "The API returned a response this UI does not understand."
            ) from exc

    def _message_for(self, response: httpx.Response) -> str:
        if response.status_code == _HTTP_UNAVAILABLE:
            return (
                "The model is still loading. Startup takes 30-40 seconds while "
                "the graph is loaded and warmed up - try again shortly."
            )
        if response.status_code == _HTTP_UNPROCESSABLE:
            return f"The API rejected this text: {_detail(response)}"
        body = response.text[:_BODY_SNIPPET_CHARS].strip()
        if not body:
            return (
                f"The API returned HTTP {response.status_code} with no details. "
                "Check that the API is running and the address is correct, "
                "then try again."
            )
        return f"The API returned HTTP {response.status_code}: {body}"

    def close(self) -> None:
        self._client.close()
