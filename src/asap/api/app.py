import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asap.api.routes import router
from asap.config import get_settings
from asap.inference.predictor import load_model, reset_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = get_settings()
    logger.info("loading model from %s", cfg.inference.model_dir)
    load_model(cfg)
    logger.info("model ready")
    yield
    reset_model()


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title=cfg.project.get("name", "arabic-sentiment-analysis"),
        version=cfg.project.get("version", "0.1.0"),
        lifespan=lifespan,
    )

    if cfg.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    """Entry point that serves the app on the configured host/port."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "asap.api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        log_level=cfg.api.log_level,
    )


if __name__ == "__main__":
    main()
