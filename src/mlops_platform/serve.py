"""Online scoring service for the ``champion`` model.

The service resolves ``models:/<name>@champion`` at startup and polls the alias;
when a promotion or rollback moves it, the new version is loaded in the
background and swapped atomically, with no restart and no dropped requests.
"""

import asyncio
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field, model_validator

from . import batch, explain
from .data import CATEGORICAL, FEATURES
from .schema import DOMAIN, Lang

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

    @model_validator(mode="after")
    def _in_domain(self) -> "Applicant":
        # Without this, any text was accepted and silently scored as the encoder's "infrequent" bucket.
        for col in CATEGORICAL:
            value = getattr(self, col)
            if value not in DOMAIN[col]:
                raise ValueError(f"{col}: {value!r} is not a credit-g category ({', '.join(DOMAIN[col])})")
        return self


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

    def current(self) -> tuple[Any, str]:
        with self._lock:
            model, version = self.model, self.version
        if model is None or version is None:
            raise RuntimeError("no model loaded")
        return model, version

    def predict(self, frame: pd.DataFrame) -> tuple[list[float], str]:
        model, version = self.current()
        return model.predict_proba(frame[FEATURES])[:, 1].tolist(), version


class PipelineRequest(BaseModel):
    """One applicant for the live pipeline. Values stay loose (text, number or null) on purpose: the
    pipeline's transform step is the one that casts, imputes and explains what it did."""

    applicant: dict[str, str | int | float | None] = Field(max_length=40)


SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def create_app(scorer: Scorer | None = None, poll_seconds: float | None = None) -> FastAPI:
    scorer = scorer or Scorer(os.getenv("MODEL_NAME", "credit-risk"))
    poll = poll_seconds if poll_seconds is not None else float(os.getenv("MODEL_POLL_SECONDS", "30"))
    threshold = float(os.getenv("DECISION_THRESHOLD", "0.5"))
    limits = batch.Limits(
        max_bytes=int(os.getenv("BATCH_MAX_BYTES", str(50 * 1024))),
        max_rows=int(os.getenv("BATCH_MAX_ROWS", "250")),
    )
    # Pipelines in flight. Each one is a few tens of ms of CPU, but the VPS is shared.
    pipelines = threading.BoundedSemaphore(int(os.getenv("PIPELINE_CONCURRENCY", "2")))

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

    @app.post("/pipeline")
    def pipeline(req: PipelineRequest, lang: Annotated[Lang, Query()] = "pt") -> StreamingResponse:
        """Every step of one prediction, as server-sent events (see explain.py). Invalid input is a 422
        before the stream starts, so the caller can refund the quota."""
        prepared, problems = explain.prepare(req.applicant, lang)
        if prepared is None:
            raise HTTPException(422, {"reason": "invalid_applicant", "problems": [p.as_dict() for p in problems]})
        try:
            model, version = scorer.current()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        if not pipelines.acquire(blocking=False):
            raise HTTPException(429, "busy")

        def stream() -> Iterator[str]:
            try:
                for event, data in explain.run(
                    model, version, prepared, model_name=scorer.model_name, threshold=threshold
                ):
                    yield sse(event, data)
                yield sse("done", {"ok": True})
            except Exception:
                logger.exception("pipeline failed")
                yield sse("error", {"message": "pipeline_failed"})
            finally:
                pipelines.release()

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    async def csv_body(request: Request) -> bytes:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in {"text/csv", "text/plain"}:
            raise HTTPException(415, "send the file as text/csv")
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limits.max_bytes:
            raise HTTPException(413, {"reason": "too_large", "max_bytes": limits.max_bytes})
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > limits.max_bytes:
                raise HTTPException(413, {"reason": "too_large", "max_bytes": limits.max_bytes})
        return body

    @app.post("/batch/validate")
    async def batch_validate(request: Request, lang: Annotated[Lang, Query()] = "pt") -> dict[str, Any]:
        """Checks a CSV without scoring it: what is wrong, by line and column, and how to fix it."""
        body = await csv_body(request)
        parsed = await asyncio.to_thread(batch.parse, body, lang, limits)
        return batch.report(parsed, limits)

    @app.post("/batch/score")
    async def batch_score(request: Request, lang: Annotated[Lang, Query()] = "pt") -> dict[str, Any]:
        """Validates again (never trusts an earlier check) and scores the valid rows. 422 when nothing can be
        scored, so the caller can refund the quota."""
        body = await csv_body(request)
        parsed = await asyncio.to_thread(batch.parse, body, lang, limits)
        summary = batch.report(parsed, limits)
        if not parsed.valid:
            raise HTTPException(422, {"reason": "no_valid_rows", "report": summary})
        try:
            model, version = scorer.current()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        results = await asyncio.to_thread(batch.score, model, parsed, threshold)
        return {
            "model_version": version,
            "threshold": threshold,
            "report": summary,
            "results": results,
            "csv": batch.to_csv(parsed, results),
        }

    admin_token = os.getenv("ADMIN_TOKEN", "")

    @app.post("/admin/reload")
    def reload(x_admin_token: Annotated[str, Header()] = "") -> dict[str, Any]:
        """Force an alias check now (e.g. right after a promotion). Disabled unless ADMIN_TOKEN is set."""
        if not admin_token or not hmac.compare_digest(x_admin_token, admin_token):
            raise HTTPException(403, "forbidden")
        return {"swapped": scorer.refresh(), "version": scorer.version}

    return app
