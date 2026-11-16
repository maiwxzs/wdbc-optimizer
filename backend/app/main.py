"""API REST (FastAPI). Los cálculos NumPy son CPU-bound: se ejecutan con
``run_in_threadpool`` para no bloquear el bucle de eventos asíncrono.

Arranque:  uvicorn app.main:app --reload --port 8000
Docs:      http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import fields

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.schemas import (
    CompareRequest,
    GradientCheckRequest,
    PredictRequest,
    StepSweepRequest,
    TrainRequest,
)
from app.services.model_service import TrainRequestDTO, get_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    service = await run_in_threadpool(get_service)  # carga WDBC + entrena el modelo por defecto
    # Precálculo en segundo plano (no retrasa el arranque): escenarios comparativos,
    # entrenamiento por defecto, curva en V y laboratorio flotante quedan en caché.
    warm = asyncio.get_running_loop().run_in_executor(None, service.warmup)
    yield
    warm.cancel()


app = FastAPI(
    title="WDBC · Motor de Optimización Numérica & CDSS Oncológico",
    version="1.0.0",
    description="Regresión logística resuelta como problema de optimización convexa sin restricciones, "
    "implementada exclusivamente con NumPy.",
    lifespan=lifespan,
)

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
)


async def _call(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/dataset/summary")
async def dataset_summary() -> dict:
    return await _call(get_service().dataset_summary)


@app.get("/api/dataset/pca")
async def dataset_pca() -> dict:
    return await _call(get_service().pca_points)


@app.get("/api/model")
async def model_info() -> dict:
    return await _call(get_service().model_info)


@app.post("/api/train")
async def train(req: TrainRequest) -> dict:
    valid = {f.name for f in fields(TrainRequestDTO)}
    dto = TrainRequestDTO(**{k: v for k, v in req.model_dump().items() if k in valid})
    return await _call(get_service().train, dto)


@app.post("/api/train/compare")
async def compare(req: CompareRequest) -> dict:
    scenarios = [s.model_dump() for s in req.scenarios] if req.scenarios else None
    return await _call(get_service().compare, scenarios, req.max_iter)


@app.post("/api/predict")
async def predict(req: PredictRequest) -> dict:
    return await _call(get_service().predict, req.sample_index, req.features)


@app.post("/api/audit/gradient-check")
async def gradient_check(req: GradientCheckRequest) -> dict:
    return await _call(get_service().gradient_check, req.point, req.h, req.seed, req.l2, req.mode)


@app.post("/api/audit/step-sweep")
async def step_sweep(req: StepSweepRequest) -> dict:
    return await _call(
        get_service().step_sweep, req.point, req.exp_min, req.exp_max, req.n, req.seed, req.l2
    )


@app.get("/api/audit/float")
async def float_audit() -> dict:
    return await _call(get_service().float_audit)
