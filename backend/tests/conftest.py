import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.loader import load_wdbc
from app.math_engine import LogisticObjective


@pytest.fixture(scope="session")
def ds():
    return load_wdbc()


@pytest.fixture()
def toy():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(64, 5))
    w_true = rng.normal(size=5)
    y = (X @ w_true + 0.3 * rng.normal(size=64) > 0).astype(float)
    return X, y


@pytest.fixture()
def wdbc_objective(ds):
    return LogisticObjective(ds.X_train, ds.y_train, l2=1e-3)


@pytest.fixture(scope="session")
def service(ds):
    from app.services.model_service import ModelService

    return ModelService(ds)


# Categoría de cada módulo de pruebas. Se asigna aquí para no duplicar marcas en
# cada archivo; permite, p. ej., `pytest -m numerical`.
_MODULE_MARKERS = {
    "test_activations_losses": ("unit", "numerical"),
    "test_gradients_fd": ("unit", "numerical"),
    "test_gradient_descent": ("unit", "numerical"),
    "test_numerical_properties": ("numerical",),
    "test_integration_pipeline": ("integration",),
    "test_service_api": ("integration",),
    "test_architecture": ("architecture",),
    "test_experiments_script": ("integration",),
}


def pytest_collection_modifyitems(items):
    for item in items:
        for name in _MODULE_MARKERS.get(item.module.__name__.rsplit(".", 1)[-1], ()):
            item.add_marker(getattr(pytest.mark, name))
