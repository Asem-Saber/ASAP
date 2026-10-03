from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from asap.config import get_settings

_api = get_settings().api

ReviewText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=_api.max_text_chars
    ),
]


class PredictRequest(BaseModel):
    text: ReviewText


class Prediction(BaseModel):
    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_dir: str
    providers: list[str]
