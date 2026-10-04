from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, task

from asap.config import get_settings

_cfg = get_settings()

def load_reviews() -> list[str]:
    path = _cfg.paths.bench_sample
    if not path.is_file():
        raise RuntimeError(f"no benchmark fixture at {path}")

    with path.open(encoding="utf-8") as handle:
        reviews = [json.loads(line)["text"] for line in handle if line.strip()]

    if not reviews:
        raise RuntimeError(f"benchmark fixture {path} is empty")
    return reviews


REVIEWS = load_reviews()
HOST = f"http://{_cfg.api.host}:{_cfg.api.port}"

PREDICT_WEIGHT = int(os.getenv("ASAP_BENCH_PREDICT_WEIGHT", "10"))
HEALTH_WEIGHT = int(os.getenv("ASAP_BENCH_HEALTH_WEIGHT", "1"))


class ServingUser(HttpUser):
    host = HOST
    wait_time = between(0.5, 2.0)

    @task(PREDICT_WEIGHT)
    def predict(self):
        with self.client.post(
            "/predict",
            json={"text": random.choice(REVIEWS)},
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("response body is not JSON")
                return
            if "sentiment" not in body:
                response.failure("response has no 'sentiment'")

    @task(HEALTH_WEIGHT)
    def health(self):
        with self.client.get("/", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif not response.json().get("model_loaded"):
                response.failure("model not loaded")
