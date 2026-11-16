"""Capa de servicio: orquesta datos, optimizador, auditoría y estado del modelo activo."""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.data.loader import WDBCDataset, load_wdbc
from app.evaluation.metrics import classification_report, roc_curve
from app.math_engine import (
    EPS_MACHINE,
    EPSILON,
    EXP_OVERFLOW_THRESHOLD,
    LogisticObjective,
    gradient_check,
    naive_sigmoid,
    sigmoid,
    step_size_sweep,
)
from app.math_engine.finite_differences import classify_relative_error
from app.optimization import GDConfig, GDResult, StopReason, gradient_descent

from .serialization import downsample_indices, to_jsonable

DEFAULT_L2 = 1e-3
DEFAULT_CONFIG = GDConfig(alpha=1.0, max_iter=8000, tol_grad=1e-5, tol_loss=1e-12)

RISK_BANDS = [
    (0.20, "bajo", "Riesgo bajo"),
    (0.50, "intermedio", "Riesgo intermedio"),
    (0.80, "alto", "Riesgo alto"),
    (1.01, "muy_alto", "Riesgo muy alto"),
]


@dataclass
class TrainRequestDTO:
    alpha: float = DEFAULT_CONFIG.alpha
    max_iter: int = DEFAULT_CONFIG.max_iter
    tol_grad: float = DEFAULT_CONFIG.tol_grad
    tol_loss: float = DEFAULT_CONFIG.tol_loss
    l2: float = DEFAULT_L2
    mode: str = "logits"
    init: str = "zeros"
    seed: int = 0
    set_active: bool = False
    max_points: int = 1500


@dataclass
class ActiveModel:
    theta: np.ndarray
    config: dict
    result_summary: dict
    test_report: dict
    train_report: dict


class ModelService:
    def __init__(self, dataset: WDBCDataset | None = None) -> None:
        self.ds = dataset or load_wdbc()
        self._lock = threading.Lock()
        self._active: ActiveModel | None = None
        self._active_version = 0
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.train(TrainRequestDTO(set_active=True))

    # ------------------------------------------------------------------ caché
    _CACHE_MAX = 64

    def _cached(self, key: tuple, compute: Callable[[], Any]) -> Any:
        """Memoriza resultados deterministas (mismos parámetros ⇒ mismo resultado).

        La clave incluye la versión del modelo activo para que lo que depende de
        θ* (init/point = "trained", PCA con probabilidades) se invalide al cambiarlo.
        """
        key = (self._active_version, *key)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        value = compute()
        with self._cache_lock:
            self._cache[key] = value
            while len(self._cache) > self._CACHE_MAX:
                self._cache.popitem(last=False)
        return value

    def warmup(self) -> None:
        """Precalcula en segundo plano lo que piden las pestañas al abrirse."""
        self.compare()
        self.train(TrainRequestDTO(max_points=800))
        self.step_sweep()
        self.float_audit()
        self.pca_points()

    # ------------------------------------------------------------------ util
    def _objective(self, l2: float = DEFAULT_L2, mode: str = "logits") -> LogisticObjective:
        if mode not in ("logits", "clipped", "naive"):
            raise ValueError("mode debe ser 'logits', 'clipped' o 'naive'")
        if l2 < 0:
            raise ValueError("l2 debe ser no negativo")
        return LogisticObjective(self.ds.X_train, self.ds.y_train, l2=l2, mode=mode)  # type: ignore[arg-type]

    def _init_theta(self, init: str, seed: int) -> np.ndarray:
        d = self.ds.n_features + 1
        if init == "zeros":
            return np.zeros(d)
        if init == "random":
            return np.random.default_rng(seed).normal(0.0, 0.5, size=d)
        if init == "trained":
            return self.active.theta.copy()
        raise ValueError("init debe ser 'zeros', 'random' o 'trained'")

    @property
    def active(self) -> ActiveModel:
        assert self._active is not None
        return self._active

    def _param_names(self) -> list[str]:
        return [*self.ds.feature_names_es, "b (sesgo)"]

    @staticmethod
    def _risk_band(p: float) -> dict:
        for upper, key, label in RISK_BANDS:
            if p < upper:
                return {"key": key, "label": label}
        return {"key": "muy_alto", "label": "Riesgo muy alto"}

    def _history_payload(self, result: GDResult, max_points: int) -> dict:
        h = result.history
        idx = downsample_indices(len(h.iteration), max_points)
        pick = lambda arr: np.asarray(arr, dtype=np.float64)[idx]  # noqa: E731
        return {
            "iteration": np.asarray(h.iteration)[idx],
            "loss": pick(h.loss),
            "grad_norm": pick(h.grad_norm),
            "loss_change": pick(h.loss_change),
            "weight_norm": pick(h.weight_norm),
            "recorded_points": len(h.iteration),
        }

    # ---------------------------------------------------------------- dataset
    def dataset_summary(self) -> dict:
        return self._cached(("summary",), self._dataset_summary)

    def _dataset_summary(self) -> dict:
        ds = self.ds
        Xr = ds.X_train_raw
        stats = []
        for j, (name, name_es) in enumerate(zip(ds.feature_names, ds.feature_names_es, strict=True)):
            col = Xr[:, j]
            stats.append(
                {
                    "index": j,
                    "name": name,
                    "name_es": name_es,
                    "group": name.split(" ")[0] if name.split(" ")[0] in ("mean", "worst") else "error",
                    "min": col.min(),
                    "max": col.max(),
                    "mean": col.mean(),
                    "std": col.std(),
                    "mean_benign": col[ds.y_train == 0].mean(),
                    "mean_malignant": col[ds.y_train == 1].mean(),
                }
            )
        y_all = np.concatenate([ds.y_train, ds.y_test])
        return to_jsonable(
            {
                "n_samples": int(y_all.size),
                "n_train": int(ds.y_train.size),
                "n_test": int(ds.y_test.size),
                "n_features": ds.n_features,
                "class_counts": {"malignant": int(y_all.sum()), "benign": int((1 - y_all).sum())},
                "label_convention": "y = 1 → maligno, y = 0 → benigno",
                "pca_explained_variance": ds.pca.explained_variance_ratio_,
                "features": stats,
            }
        )

    def pca_points(self) -> dict:
        return self._cached(("pca",), self._pca_points)

    def _pca_points(self) -> dict:
        ds = self.ds
        theta = self.active.theta
        p_test = sigmoid(ds.X_test @ theta[:-1] + theta[-1])
        return to_jsonable(
            {
                "train": [
                    {"x": p[0], "y": p[1], "label": int(lbl)} for p, lbl in zip(ds.pca_train, ds.y_train, strict=True)
                ],
                "test": [
                    {
                        "i": i,
                        "x": p[0],
                        "y": p[1],
                        "label": int(lbl),
                        "proba": pr,
                        "pred": int(pr >= 0.5),
                    }
                    for i, (p, lbl, pr) in enumerate(zip(ds.pca_test, ds.y_test, p_test, strict=True))
                ],
                "explained_variance": ds.pca.explained_variance_ratio_,
            }
        )

    # ---------------------------------------------------------------- training
    def train(self, req: TrainRequestDTO) -> dict:
        # Sin efectos secundarios (no promueve el modelo) ⇒ resultado cacheable.
        if not req.set_active and self._active is not None:
            return self._cached(("train", *asdict(req).values()), lambda: self._train(req))
        return self._train(req)

    def _train(self, req: TrainRequestDTO) -> dict:
        cfg = GDConfig(alpha=req.alpha, max_iter=req.max_iter, tol_grad=req.tol_grad, tol_loss=req.tol_loss)
        obj = self._objective(req.l2, req.mode)
        theta0 = self._init_theta(req.init, req.seed) if self._active or req.init != "trained" else np.zeros(obj.dim)
        result = gradient_descent(obj, theta0, cfg)

        L = obj.lipschitz_constant()
        theta = result.theta
        with np.errstate(all="ignore"):
            p_test = sigmoid(self.ds.X_test @ theta[:-1] + theta[-1])
            p_train = sigmoid(self.ds.X_train @ theta[:-1] + theta[-1])
        test_report = classification_report(self.ds.y_test, p_test)
        train_report = classification_report(self.ds.y_train, p_train)

        names = self._param_names()
        order = np.argsort(-np.abs(np.nan_to_num(theta[:-1])))
        weights = [{"name": names[j], "index": int(j), "weight": theta[j]} for j in order]

        summary = {
            "config": {**asdict(cfg), "l2": req.l2, "mode": req.mode, "init": req.init},
            "stop_reason": result.stop_reason,
            "message": result.message,
            "converged": result.converged,
            "diverged": result.stop_reason == StopReason.DIVERGENCE,
            "divergence_detail": result.divergence_detail,
            "iterations": result.iterations,
            "elapsed_ms": result.elapsed_ms,
            "final_loss": result.final_loss,
            "final_grad_norm": result.final_grad_norm,
            "numeric_flags": result.numeric_flags,
            "lipschitz_L": L,
            "alpha_critical": 2.0 / L,
            "alpha_over_critical": req.alpha / (2.0 / L),
            "bias": theta[-1],
        }
        active_updated = False
        if req.set_active and result.stop_reason != StopReason.DIVERGENCE:
            with self._lock:
                self._active = ActiveModel(
                    theta=theta.copy(),
                    config=summary["config"],
                    result_summary=summary,
                    test_report=test_report,
                    train_report=train_report,
                )
                self._active_version += 1
            active_updated = True

        return to_jsonable(
            {
                **summary,
                "history": self._history_payload(result, req.max_points),
                "metrics": {"train": train_report, "test": test_report},
                "roc_test": roc_curve(self.ds.y_test, p_test),
                "weights": weights,
                "active_updated": active_updated,
            }
        )

    def compare(self, scenarios: list[dict] | None = None, max_iter: int = 3000) -> dict:
        if scenarios is None:
            return self._cached(("compare", max_iter), lambda: self._compare(None, max_iter))
        return self._compare(scenarios, max_iter)

    def _compare(self, scenarios: list[dict] | None, max_iter: int) -> dict:
        L = self._objective().lipschitz_constant()
        presets = scenarios or [
            {"label": "Conservador α=0.05", "alpha": 0.05, "mode": "logits", "l2": DEFAULT_L2},
            {"label": "Teórico α=1/L", "alpha": 1.0 / L, "mode": "logits", "l2": DEFAULT_L2},
            {"label": "Agresivo α=5", "alpha": 5.0, "mode": "logits", "l2": DEFAULT_L2},
            {"label": "Oscilatorio α=100", "alpha": 100.0, "mode": "logits", "l2": DEFAULT_L2},
            {"label": "Divergente α=1e5", "alpha": 1e5, "mode": "logits", "l2": DEFAULT_L2},
            {"label": "Ingenuo α=1 (sin estabilizar)", "alpha": 1.0, "mode": "naive", "l2": DEFAULT_L2},
        ]
        runs = []
        for sc in presets:
            obj = self._objective(sc.get("l2", DEFAULT_L2), sc.get("mode", "logits"))
            res = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=sc["alpha"], max_iter=max_iter))
            with np.errstate(all="ignore"):
                p = sigmoid(self.ds.X_test @ res.theta[:-1] + res.theta[-1])
            rep = classification_report(self.ds.y_test, p)
            runs.append(
                {
                    **sc,
                    "stop_reason": res.stop_reason,
                    "message": res.message,
                    "divergence_detail": res.divergence_detail,
                    "iterations": res.iterations,
                    "final_loss": res.final_loss,
                    "final_grad_norm": res.final_grad_norm,
                    "test_accuracy": rep["accuracy"],
                    "alpha_over_critical": sc["alpha"] / (2.0 / L),
                    "numeric_flags": res.numeric_flags,
                    "history": self._history_payload(res, 400),
                }
            )
        return to_jsonable({"lipschitz_L": L, "alpha_critical": 2.0 / L, "max_iter": max_iter, "runs": runs})

    # --------------------------------------------------------------- inference
    def model_info(self) -> dict:
        a = self.active
        names = self._param_names()
        order = np.argsort(-np.abs(a.theta[:-1]))
        return to_jsonable(
            {
                "summary": a.result_summary,
                "metrics": {"train": a.train_report, "test": a.test_report},
                "weights": [{"name": names[j], "index": int(j), "weight": a.theta[j]} for j in order],
                "bias": a.theta[-1],
            }
        )

    def predict(self, sample_index: int | None = None, features: list[float] | None = None) -> dict:
        ds = self.ds
        true_label = None
        if features is not None:
            x_raw = np.asarray(features, dtype=np.float64)
            if x_raw.shape != (ds.n_features,):
                raise ValueError(f"se esperaban {ds.n_features} características, llegaron {x_raw.size}")
            if not np.all(np.isfinite(x_raw)):
                raise ValueError("las características deben ser finitas")
            x_std = ds.standardize(x_raw)[0]
        elif sample_index is not None:
            if not 0 <= sample_index < ds.X_test.shape[0]:
                raise ValueError(f"sample_index fuera de rango [0, {ds.X_test.shape[0] - 1}]")
            x_raw = ds.X_test_raw[sample_index]
            x_std = ds.X_test[sample_index]
            true_label = int(ds.y_test[sample_index])
        else:
            raise ValueError("indique sample_index o features")

        theta = self.active.theta
        w, b = theta[:-1], float(theta[-1])
        contrib = w * x_std
        z = float(contrib.sum() + b)
        p = float(sigmoid(z))
        names = ds.feature_names_es
        order = np.argsort(-np.abs(contrib))
        pc = ds.project(x_std)[0]
        return to_jsonable(
            {
                "probability_malignant": p,
                "logit": z,
                "prediction": int(p >= 0.5),
                "prediction_label": "Maligno" if p >= 0.5 else "Benigno",
                "risk_band": self._risk_band(p),
                "true_label": true_label,
                "bias": b,
                "pca": {"x": pc[0], "y": pc[1]},
                "raw_features": x_raw,
                "contributions": [
                    {
                        "name": names[j],
                        "index": int(j),
                        "raw_value": x_raw[j],
                        "std_value": x_std[j],
                        "weight": w[j],
                        "contribution": contrib[j],
                    }
                    for j in order
                ],
            }
        )

    # ------------------------------------------------------------------ audit
    def _audit_point(self, point: str, seed: int, l2: float) -> np.ndarray:
        if point == "trained":
            return self.active.theta.copy()
        return self._init_theta(point, seed)

    def gradient_check(self, point: str = "random", h: float = 1e-5, seed: int = 0, l2: float = DEFAULT_L2, mode: str = "logits") -> dict:
        obj = self._objective(l2, mode)
        theta = self._audit_point(point, seed, l2)
        res = gradient_check(obj.loss, obj.grad, theta, h)
        names = self._param_names()
        comps = [
            {
                "name": names[j],
                "analytic": res["analytic"][j],
                "numeric": res["numeric"][j],
                "abs_error": res["abs_error"][j],
                "rel_error": res["component_rel_error"][j],
            }
            for j in range(theta.size)
        ]
        return to_jsonable(
            {
                "point": point,
                "h": h,
                "loss_at_point": obj.loss(theta),
                "relative_error": res["relative_error"],
                "max_abs_error": res["max_abs_error"],
                "analytic_norm": res["analytic_norm"],
                "numeric_norm": res["numeric_norm"],
                "verdict": res["verdict"],
                "message": res["message"],
                "loss_evaluations": 2 * theta.size,
                "components": comps,
            }
        )

    def step_sweep(
        self, point: str = "random", exp_min: float = -16, exp_max: float = 0, n: int = 49, seed: int = 0, l2: float = DEFAULT_L2
    ) -> dict:
        if not (exp_min < exp_max and 5 <= n <= 200):
            raise ValueError("rango de h inválido o n fuera de [5, 200]")
        return self._cached(
            ("sweep", point, exp_min, exp_max, n, seed, l2),
            lambda: self._step_sweep(point, exp_min, exp_max, n, seed, l2),
        )

    def _step_sweep(self, point: str, exp_min: float, exp_max: float, n: int, seed: int, l2: float) -> dict:
        obj = self._objective(l2, "logits")
        theta = self._audit_point(point, seed, l2)
        hs = np.logspace(exp_min, exp_max, n)
        res = step_size_sweep(obj.loss, obj.grad, theta, hs)
        best_c = res["central_best"]["error"]
        res["verdict_at_best"] = classify_relative_error(best_c)[0]
        res["point"] = point
        res["loss_at_point"] = obj.loss(theta)
        return to_jsonable(res)

    def float_audit(self) -> dict:
        return self._cached(("float",), self._float_audit)

    def _float_audit(self) -> dict:
        """Compara sigmoide/pérdida ingenua, acotada (clip) y Log-Sum-Exp."""
        z_grid = np.concatenate([np.linspace(-800, -50, 151), np.linspace(-50, 60, 221)])
        with np.errstate(all="ignore"):
            s_naive = naive_sigmoid(z_grid)
            s_stable = sigmoid(z_grid)
            # pérdida para y = 1: ℓ = -log σ(z)
            loss_exact = np.logaddexp(0.0, -z_grid)
            loss_naive = -np.log(s_naive)
            loss_clip = -np.log(np.clip(s_stable, EPSILON, 1 - EPSILON))
            # pérdida para y = 0: ℓ = -log(1 - σ(z))
            loss0_exact = np.logaddexp(0.0, z_grid)
            loss0_naive = -np.log(1.0 - s_naive)

        def ieee(v: float):
            """Valores especiales IEEE-754 como texto (JSON no admite inf/NaN)."""
            v = float(v)
            if np.isnan(v):
                return "NaN"
            if np.isinf(v):
                return "+inf" if v > 0 else "-inf"
            return v

        probes = [-1000.0, -745.2, -710.0, -40.0, -20.0, 0.0, 20.0, 36.0, 37.5, 40.0, 710.0]
        table = []
        with np.errstate(all="ignore"):
            for z in probes:
                sn, ss = float(naive_sigmoid(z)), float(sigmoid(z))
                table.append(
                    {
                        "z": z,
                        "sigmoid_naive": ieee(sn),
                        "sigmoid_stable": ieee(ss),
                        "exp_overflow": bool(-z > EXP_OVERFLOW_THRESHOLD),
                        "loss_y1_naive": ieee(float(-np.log(sn))),
                        "loss_y1_clip": ieee(float(-np.log(np.clip(ss, EPSILON, 1 - EPSILON)))),
                        "loss_y1_exact": ieee(float(np.logaddexp(0.0, -z))),
                        "loss_y0_naive": ieee(float(-np.log(1.0 - sn))),
                        "loss_y0_exact": ieee(float(np.logaddexp(0.0, z))),
                    }
                )
        fi = np.finfo(np.float64)
        return to_jsonable(
            {
                "z": z_grid,
                "sigmoid_naive": s_naive,
                "sigmoid_stable": s_stable,
                "loss_y1": {"exact": loss_exact, "naive": loss_naive, "clipped": loss_clip},
                "loss_y0": {"exact": loss0_exact, "naive": loss0_naive},
                "table": table,
                "constants": {
                    "eps_machine": EPS_MACHINE,
                    "clip_epsilon": EPSILON,
                    "clip_ceiling": -np.log(EPSILON),
                    "float_max": fi.max,
                    "float_tiny": fi.tiny,
                    "exp_overflow_threshold": EXP_OVERFLOW_THRESHOLD,
                    "sigmoid_saturation_z": float(-np.log(EPS_MACHINE / 2)),
                    "underflow_threshold": float(np.log(fi.smallest_subnormal)),
                },
                "notes": [
                    "exp(x) desborda a +inf para x > ln(DBL_MAX) ≈ 709.78.",
                    "σ(z) redondea a 1.0 exactamente para z ≳ 36.7 (1 + e^{-z} = 1 en float64); log(1-σ) = -inf.",
                    "0·log(0) = 0·(-inf) = NaN: origen típico de NaN en la BCE ingenua.",
                    "El clip con ε = 1e-15 acota la pérdida por muestra en ≈ 34.54, sesgando el gradiente de muestras muy mal clasificadas.",
                    "La formulación Log-Sum-Exp ℓ = log(1+e^z) - y·z es exacta para todo z representable.",
                ],
            }
        )


_service: ModelService | None = None
_service_lock = threading.Lock()


def get_service() -> ModelService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ModelService()
    return _service
