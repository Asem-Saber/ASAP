FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

FROM base AS builder-api
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra serve
COPY src/ ./src/
RUN uv sync --frozen --extra serve

FROM base AS api
COPY --from=builder-api --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

ENV ASAP_API__HOST=0.0.0.0
USER appuser
EXPOSE 8000
CMD ["asap-api"]

FROM base AS builder-ui
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra ui
COPY src/ ./src/
RUN uv sync --frozen --extra ui

FROM base AS ui
COPY --from=builder-ui --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
USER appuser
EXPOSE 8501
CMD ["asap-ui"]
