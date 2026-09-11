from typing import Annotated

from fastapi import Depends, HTTPException, status
from asap.inference import predictor
from asap.inference.predictor import SentimentModel


def get_model() -> SentimentModel:
    if predictor.sentiment_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model is not loaded",
        )
    return predictor.sentiment_model


ModelDep = Annotated[SentimentModel, Depends(get_model)]
