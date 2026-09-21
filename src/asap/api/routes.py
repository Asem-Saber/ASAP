import anyio
from fastapi import APIRouter

from asap.api.deps import ModelDep
from asap.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    Prediction,
)
from asap.config import get_settings
from asap.inference import predictor

router = APIRouter()

_inference_limiter = anyio.CapacityLimiter(1)


@router.get("/", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    cfg = get_settings()
    model = predictor.sentiment_model

    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        model_dir=str(cfg.inference.model_dir),
        providers=list(model.providers) if model is not None else [cfg.inference.provider],
    )


@router.post("/predict", response_model=Prediction, tags=["predict"])
async def predict(request: PredictRequest, model: ModelDep) -> Prediction:
    result = await anyio.to_thread.run_sync(
        model.predict, request.text, limiter=_inference_limiter
    )
    return Prediction(**result)


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["predict"])
async def predict_batch(
    request: BatchPredictRequest, model: ModelDep
) -> BatchPredictResponse:
    results = await anyio.to_thread.run_sync(
        model.predict_batch, request.texts, limiter=_inference_limiter
    )
    return BatchPredictResponse(predictions=[Prediction(**r) for r in results])
