"""Hace cumplir la restricción del proyecto: el motor numérico es NumPy puro."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
FORBIDDEN = {"sklearn", "torch", "tensorflow", "jax", "scipy"}
PURE_PACKAGES = ["math_engine", "optimization", "evaluation"]
ALLOWED_SKLEARN = {"load_breast_cancer", "train_test_split", "StandardScaler", "PCA"}


def _imports(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, [a.name for a in node.names]))
    return out


def test_numeric_core_has_no_ml_libraries():
    for pkg in PURE_PACKAGES:
        for f in (APP / pkg).rglob("*.py"):
            for mod, _ in _imports(f):
                assert mod.split(".")[0] not in FORBIDDEN, f"{f.name} importa {mod}"


def test_sklearn_only_for_allowed_utilities():
    for f in APP.rglob("*.py"):
        for mod, names in _imports(f):
            if mod.split(".")[0] == "sklearn":
                assert f.name == "loader.py", f"sklearn usado fuera del cargador: {f}"
                assert set(names) <= ALLOWED_SKLEARN, f"uso no permitido de sklearn: {names}"
