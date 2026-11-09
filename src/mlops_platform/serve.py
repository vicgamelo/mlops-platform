"""Online scoring service for the ``champion`` model.

The service resolves ``models:/<name>@champion`` at startup and polls the alias;
when a promotion or rollback moves it, the new version is loaded in the
background and swapped atomically, with no restart and no dropped requests.
"""

import asyncio
import hmac
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from .data import FEATURES

logger = logging.getLogger("mlops_platform.serve")


class Applicant(BaseModel):
    checking_status: Literal["<0", "0<=X<200", ">=200", "no checking"]
    duration: Annotated[int, Field(ge=1, le=120)]
    credit_history: str
    purpose: str
    credit_amount: Annotated[float, Field(gt=0, le=1_000_000)]
    savings_status: str
    employment: str
    installment_commitment: Annotated[int, Field(ge=1, le=4)]
    other_parties: str
    residence_since: Annotated[int, Field(ge=1, le=4)]
    property_magnitude: str
    age: Annotated[int, Field(ge=18, le=100)]
    other_payment_plans: str
    housing: str
    existing_credits: Annotated[int, Field(ge=0, le=10)]
    job: str
    num_dependents: Annotated[int, Field(ge=0, le=10)]
    own_telephone: Literal["none", "yes"]


class PredictRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


class Scorer:
    def __init__(self, model_name: str, alias: str = "champion") -> None:
        self.model_name, self.alias = model_name, alias
        self.client = MlflowClient()
        self.model: Any = None
        self.version: str | None = None
        self.loaded_at: float | None = None
        self._lock = threading.Lock()

    def refresh(self) -> bool:
        """Load the aliased version if it changed. Returns True when swapped."""
        version = str(self.client.get_model_version_by_alias(self.model_name, self.alias).version)
        if version == self.version:
            return False
        model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")
        with self._lock:
            self.model, self.version, self.loaded_at = model, version, time.time()
        logger.info("serving %s version %s", self.model_name, version)
        return True

    def predict(self, frame: pd.DataFrame) -> tuple[list[float], str]:
        with self._lock:
            model, version = self.model, self.version
        if model is None or version is None:
            raise RuntimeError("no model loaded")
        return model.predict_proba(frame[FEATURES])[:, 1].tolist(), version


def create_app(scorer: Scorer | None = None, poll_seconds: float | None = None) -> FastAPI:
    scorer = scorer or Scorer(os.getenv("MODEL_NAME", "credit-risk"))
    poll = poll_seconds if poll_seconds is not None else float(os.getenv("MODEL_POLL_SECONDS", "30"))
    threshold = float(os.getenv("DECISION_THRESHOLD", "0.5"))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(scorer.refresh)

        async def watch() -> None:
            while True:
                await asyncio.sleep(poll)
                try:
                    await asyncio.to_thread(scorer.refresh)
                except Exception:
                    logger.exception("alias refresh failed; keeping version %s", scorer.version)

        task = asyncio.create_task(watch()) if poll > 0 else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="credit-risk scoring", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok" if scorer.model is not None else "loading", "model_version": scorer.version}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        return {
            "name": scorer.model_name,
            "alias": scorer.alias,
            "version": scorer.version,
            "loaded_at": scorer.loaded_at,
        }

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        frame = pd.DataFrame([a.model_dump() for a in req.applicants])
        try:
            scores, version = scorer.predict(frame)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "model_version": version,
            "predictions": [
                {"default_probability": round(p, 6), "decision": "review" if p >= threshold else "approve"}
                for p in scores
            ],
        }

    admin_token = os.getenv("ADMIN_TOKEN", "")

    @app.post("/admin/reload")
    def reload(x_admin_token: Annotated[str, Header()] = "") -> dict[str, Any]:
        """Force an alias check now (e.g. right after a promotion). Disabled unless ADMIN_TOKEN is set."""
        if not admin_token or not hmac.compare_digest(x_admin_token, admin_token):
            raise HTTPException(403, "forbidden")
        return {"swapped": scorer.refresh(), "version": scorer.version}

    return app
