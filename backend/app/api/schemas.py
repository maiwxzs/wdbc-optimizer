"""Esquemas Pydantic v2 de las peticiones REST (las respuestas son dicts JSON-safe)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["logits", "clipped", "naive"]
Init = Literal["zeros", "random", "trained"]
Point = Literal["zeros", "random", "trained"]


class TrainRequest(BaseModel):
    alpha: float = Field(1.0, gt=0, le=1e12, description="Tasa de aprendizaje α")
    max_iter: int = Field(8000, ge=1, le=100_000)
    tol_grad: float = Field(1e-5, ge=0)
    tol_loss: float = Field(1e-12, ge=0)
    l2: float = Field(1e-3, ge=0, le=100)
    mode: Mode = "logits"
    init: Init = "zeros"
    seed: int = 0
    set_active: bool = False
    max_points: int = Field(1500, ge=50, le=10_000)


class Scenario(BaseModel):
    label: str
    alpha: float = Field(..., gt=0, le=1e12)
    mode: Mode = "logits"
    l2: float = Field(1e-3, ge=0)


class CompareRequest(BaseModel):
    scenarios: list[Scenario] | None = None
    max_iter: int = Field(3000, ge=10, le=20_000)


class PredictRequest(BaseModel):
    sample_index: int | None = Field(None, ge=0)
    features: list[float] | None = Field(None, min_length=30, max_length=30)

    @model_validator(mode="after")
    def _one_of(self) -> "PredictRequest":
        if (self.sample_index is None) == (self.features is None):
            raise ValueError("indique exactamente uno: sample_index o features")
        return self


class GradientCheckRequest(BaseModel):
    point: Point = "random"
    h: float = Field(1e-5, gt=0, le=1)
    seed: int = 0
    l2: float = Field(1e-3, ge=0)
    mode: Mode = "logits"


class StepSweepRequest(BaseModel):
    point: Point = "random"
    exp_min: float = Field(-16, ge=-20, le=-1)
    exp_max: float = Field(0, ge=-10, le=1)
    n: int = Field(49, ge=5, le=200)
    seed: int = 0
    l2: float = Field(1e-3, ge=0)
