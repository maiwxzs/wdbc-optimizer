"""Reproduce los experimentos documentados en docs/experiments/results.md.

Uso (desde la raíz del repositorio):

    python scripts/reproduce_experiments.py                 # métricas + figuras
    python scripts/reproduce_experiments.py --no-figures    # solo métricas (sin matplotlib)
    python scripts/reproduce_experiments.py --output otra/carpeta

Genera:
    results/metrics/*.json   valores numéricos (JSON estricto: NaN/Inf → null)
    results/figures/*.png    figuras (requiere matplotlib)

Todos los experimentos son deterministas: partición con semilla 42, inicialización
θ⁽⁰⁾ = 0 (o semilla explícita cuando es aleatoria). Los tiempos de ejecución sí
dependen de la máquina y se registran solo como referencia.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from app.data.loader import load_wdbc  # noqa: E402
from app.evaluation.metrics import classification_report, roc_curve  # noqa: E402
from app.math_engine import LogisticObjective, gradient_check, sigmoid, step_size_sweep  # noqa: E402
from app.optimization import GDConfig, gradient_descent  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402
from app.services.serialization import to_jsonable  # noqa: E402

L2 = 1e-3
SEED_SPLIT = 42

# Paleta categórica validada (orden fijo, sin ciclar) — ver docs/experiments/results.md
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- experimentos


def environment() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "split_seed": SEED_SPLIT,
        "l2": L2,
    }


def exp_default_model(ds) -> tuple[dict, dict]:
    """E1 · Modelo por defecto: α = 1, θ⁽⁰⁾ = 0, tol_grad = 1e-5, tol_loss = 1e-12."""
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=L2)
    cfg = GDConfig(alpha=1.0, max_iter=8000, tol_grad=1e-5, tol_loss=1e-12)
    res = gradient_descent(obj, np.zeros(obj.dim), cfg)
    p_test = sigmoid(ds.X_test @ res.w + res.b)
    p_train = sigmoid(ds.X_train @ res.w + res.b)
    L = obj.lipschitz_constant()
    out = {
        "config": {"alpha": 1.0, "max_iter": 8000, "tol_grad": 1e-5, "tol_loss": 1e-12, "l2": L2, "init": "zeros"},
        "lipschitz_L": L,
        "alpha_critical_2_over_L": 2.0 / L,
        "stop_reason": res.stop_reason.value,
        "iterations": res.iterations,
        "final_loss": res.final_loss,
        "final_grad_norm": res.final_grad_norm,
        "elapsed_ms_reference_only": res.elapsed_ms,
        "metrics_train": classification_report(ds.y_train, p_train),
        "metrics_test": classification_report(ds.y_test, p_test),
        "bias": res.b,
    }
    curves = {
        "history": {"iteration": res.history.iteration, "loss": res.history.loss, "grad_norm": res.history.grad_norm},
        "roc": roc_curve(ds.y_test, p_test),
    }
    return out, curves


def exp_scenarios(service: ModelService) -> dict:
    """E2 · Escenarios comparativos (mismos presets que la pestaña de Telemetría)."""
    return service.compare(max_iter=3000)


def exp_learning_rate_sweep(ds) -> dict:
    """E3 · Barrido de α en escala logarítmica: iteraciones hasta converger y estado final."""
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=L2)
    L = obj.lipschitz_constant()
    rows = []
    for alpha in np.logspace(-2, 5, 29):
        r = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=float(alpha), max_iter=5000))
        p = sigmoid(ds.X_test @ r.w + r.b)
        rows.append(
            {
                "alpha": float(alpha),
                "alpha_over_critical": float(alpha * L / 2.0),
                "stop_reason": r.stop_reason.value,
                "iterations": r.iterations,
                "final_loss": r.final_loss,
                "final_grad_norm": r.final_grad_norm,
                "test_accuracy": classification_report(ds.y_test, p)["accuracy"],
            }
        )
    return {"max_iter": 5000, "lipschitz_L": L, "alpha_critical": 2.0 / L, "runs": rows}


def exp_regularization(ds) -> dict:
    """E4 · Efecto de λ: con λ = 0 el conjunto es casi separable y ‖w‖ crece sin cota."""
    out = {}
    for lam in (0.0, 1e-3):
        obj = LogisticObjective(ds.X_train, ds.y_train, l2=lam)
        r = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=5.0, max_iter=4000, tol_grad=0.0, tol_loss=0.0))
        p = sigmoid(ds.X_test @ r.w + r.b)
        out[f"l2={lam:g}"] = {
            "final_loss": r.final_loss,
            "final_grad_norm": r.final_grad_norm,
            "final_weight_norm": r.history.weight_norm[-1],
            "test_accuracy": classification_report(ds.y_test, p)["accuracy"],
            "weight_norm_curve": r.history.weight_norm[::20],
            "iteration_curve": r.history.iteration[::20],
        }
    return {"alpha": 5.0, "max_iter": 4000, "runs": out}


def exp_gradient_audit(ds, theta_trained: np.ndarray) -> dict:
    """E5 · Chequeo de gradiente (h = 1e-5) y barrido de h (curva en V)."""
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=L2)
    theta_rand = np.random.default_rng(0).normal(0.0, 0.5, obj.dim)
    checks = {}
    for name, theta in (("random_seed0", theta_rand), ("zeros", np.zeros(obj.dim)), ("trained", theta_trained)):
        r = gradient_check(obj.loss, obj.grad, theta, h=1e-5)
        checks[name] = {
            "relative_error": r["relative_error"],
            "verdict": r["verdict"],
            "analytic_norm": r["analytic_norm"],
            "max_abs_error": r["max_abs_error"],
        }
    sweep = step_size_sweep(obj.loss, obj.grad, theta_rand, np.logspace(-16, 0, 49))
    return {"gradient_check_h": 1e-5, "checks": checks, "step_sweep_random_seed0": sweep}


def exp_float(service: ModelService) -> dict:
    """E6 · Laboratorio IEEE-754 (sigmoide/pérdida ingenua vs clip vs Log-Sum-Exp)."""
    return service.float_audit()


# --------------------------------------------------------------------- figuras


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def make_figures(fig_dir: Path, default_curves: dict, scenarios: dict, lr: dict, reg: dict, audit: dict, flt: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(fig_dir / name)
        plt.close(fig)
        written.append(name)

    # F1 · Modelo por defecto: J y ‖∇J‖ (dos paneles, una escala por panel)
    h = default_curves["history"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    J = np.asarray(h["loss"])
    a1.plot(h["iteration"], J - J[-1] + 1e-16, color=PALETTE[0])
    a1.set_yscale("log")
    a1.set_title("J(θ⁽ᵏ⁾) − J_final  ·  α = 1")
    a1.set_xlabel("iteración k")
    a2.plot(h["iteration"], h["grad_norm"], color=PALETTE[0])
    a2.axhline(1e-5, color=MUTED, ls="--", lw=1)
    a2.text(h["iteration"][-1], 1.4e-5, "tol_grad = 1e-5", ha="right", color=MUTED, fontsize=8)
    a2.set_yscale("log")
    a2.set_title("‖∇J(θ⁽ᵏ⁾)‖₂")
    a2.set_xlabel("iteración k")
    save(fig, "default_model_convergence.png")

    # F2 · Escenarios: J(θ⁽ᵏ⁾) por escenario
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for color, run in zip(PALETTE, scenarios["runs"], strict=False):
        it = np.asarray(run["history"]["iteration"], dtype=float)
        loss = np.array([np.nan if v is None else v for v in run["history"]["loss"]], dtype=float)
        ax.plot(it, loss, color=color, label=f"{run['label']} → {run['stop_reason']}")
        last = np.where(np.isfinite(loss))[0]
        if last.size:
            ax.plot(it[last[-1]], loss[last[-1]], "o", ms=6, color=color, mec="#fcfcfb", mew=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("iteración k")
    ax.set_ylabel("J(θ⁽ᵏ⁾)")
    ax.set_title("Escenarios de tasa de aprendizaje (k_max = 3000)")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "scenarios_loss.png")

    # F3 · Barrido de α: iteraciones (panel 1) y J final (panel 2)
    runs = lr["runs"]
    alphas = np.array([r["alpha"] for r in runs])
    its = np.array([r["iterations"] for r in runs])
    jf = np.array([np.nan if r["final_loss"] is None else r["final_loss"] for r in runs])
    stop = [r["stop_reason"] for r in runs]
    marker = {"grad_norm": ("o", PALETTE[0], "converge (‖∇J‖)"), "loss_change": ("s", PALETTE[2], "converge (|ΔJ|)"),
              "max_iter": ("^", PALETTE[3], "k_max"), "divergence": ("X", PALETTE[1], "divergencia")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, y, title in ((a1, its, "Iteraciones ejecutadas"), (a2, jf, "J final")):
        ax.plot(alphas, y, color=GRID, lw=1.2, zorder=1)
        for key, (mk, col, lab) in marker.items():
            sel = np.array([s == key for s in stop])
            if sel.any():
                ax.scatter(alphas[sel], y[sel], marker=mk, s=40, color=col, edgecolor="#fcfcfb", lw=1, label=lab, zorder=3)
        ax.axvline(lr["alpha_critical"], color=MUTED, ls="--", lw=1)
        ax.text(lr["alpha_critical"] * 1.15, ax.get_ylim()[1] if ax is a1 else np.nanmax(y), "2/L", color=MUTED, fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xlabel("α")
        ax.set_title(title)
    a2.set_yscale("log")
    a1.legend(fontsize=8, loc="center left")
    save(fig, "learning_rate_sweep.png")

    # F4 · Regularización: ‖w‖ vs k
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for color, (name, r) in zip((PALETTE[1], PALETTE[0]), reg["runs"].items(), strict=True):
        ax.plot(r["iteration_curve"], r["weight_norm_curve"], color=color, label=f"λ = {name.split('=')[1]}")
    ax.set_xlabel("iteración k")
    ax.set_ylabel("‖w⁽ᵏ⁾‖₂")
    ax.set_title("Norma de los pesos con y sin regularización (α = 5)")
    ax.legend()
    save(fig, "regularization_weight_norm.png")

    # F5 · Curva en V
    sw = audit["step_sweep_random_seed0"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.loglog(sw["h"], sw["central_error"], "o-", ms=4, color=PALETTE[0], label="central  O(h²)")
    ax.loglog(sw["h"], sw["forward_error"], "s-", ms=4, color=PALETTE[1], label="progresiva  O(h)")
    ax.axvline(sw["central_h_opt_theory"], color=PALETTE[0], ls=":", lw=1)
    ax.axvline(sw["forward_h_opt_theory"], color=PALETTE[1], ls=":", lw=1)
    ax.text(sw["central_h_opt_theory"] * 1.3, 2e-1, r"$\varepsilon^{1/3}$", color=MUTED, fontsize=8)
    ax.text(sw["forward_h_opt_theory"] * 1.3, 2e-1, r"$\sqrt{\varepsilon}$", color=MUTED, fontsize=8)
    ax.set_xlabel("paso h")
    ax.set_ylabel("error relativo  ‖gₐ − gₙ‖ / (‖gₐ‖ + ‖gₙ‖)")
    ax.set_title("Error de diferencias finitas vs h (θ aleatorio, semilla 0)")
    ax.legend()
    save(fig, "finite_difference_v_curve.png")

    # F6 · Pérdida por muestra (y = 1): exacta vs clip vs ingenua
    z = np.asarray(flt["z"], dtype=float)
    ly = flt["loss_y1"]
    conv = lambda a: np.array([np.nan if v is None else v for v in a], dtype=float)  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(z, conv(ly["exact"]), color=PALETTE[0], label="Log-Sum-Exp (exacta)")
    ax.plot(z, conv(ly["clipped"]), color=PALETTE[1], ls="--", label="clip ε = 1e-15 (techo ≈ 34.54)")
    ax.plot(z, conv(ly["naive"]), color=PALETTE[2], ls=":", lw=2.5, label="ingenua (+inf en la zona gris; 0 si z ≳ 36.7)")
    ax.axvspan(z.min(), -flt["constants"]["exp_overflow_threshold"], color=GRID, alpha=0.6, lw=0)
    ax.text(-760, 1e-20, "exp(−z) desborda:\nla forma ingenua\nvale +inf", color=MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("logit z")
    ax.set_ylabel("ℓ(z, y = 1) = −log σ(z)")
    ax.set_title("Pérdida por muestra en tres formulaciones")
    ax.legend()
    save(fig, "float_loss_formulations.png")

    # F7 · ROC en test
    roc = default_curves["roc"]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.plot(roc["fpr"], roc["tpr"], color=PALETTE[0])
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("sensibilidad")
    ax.set_title("ROC en test (modelo por defecto)")
    ax.set_aspect("equal")
    save(fig, "roc_test.png")

    return written


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    ds = load_wdbc(test_size=0.2, random_state=SEED_SPLIT)
    service = ModelService(ds)

    default, curves = exp_default_model(ds)
    obj = LogisticObjective(ds.X_train, ds.y_train, l2=L2)
    theta_trained = gradient_descent(obj, np.zeros(obj.dim), GDConfig(alpha=1.0, max_iter=8000)).theta
    results = {
        "environment": environment(),
        "e1_default_model": default,
        "e2_scenarios": exp_scenarios(service),
        "e3_learning_rate_sweep": exp_learning_rate_sweep(ds),
        "e4_regularization": exp_regularization(ds),
        "e5_gradient_audit": exp_gradient_audit(ds, theta_trained),
        "e6_float_audit": exp_float(service),
    }

    metrics_dir = args.output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for key, value in results.items():
        (metrics_dir / f"{key}.json").write_text(
            json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

    figures: list[str] = []
    if not args.no_figures:
        figures = make_figures(
            args.output / "figures",
            to_jsonable(curves),
            to_jsonable(results["e2_scenarios"]),
            to_jsonable(results["e3_learning_rate_sweep"]),
            to_jsonable(results["e4_regularization"]),
            to_jsonable(results["e5_gradient_audit"]),
            to_jsonable(results["e6_float_audit"]),
        )

    print(f"Métricas: {metrics_dir}  ({len(results)} archivos)")
    if figures:
        print(f"Figuras:  {args.output / 'figures'}  ({len(figures)} archivos)")
    print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
    return results


if __name__ == "__main__":
    main()
