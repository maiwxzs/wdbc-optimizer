import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.loader import load_wdbc  # noqa: E402
from app.math_engine import LogisticObjective  # noqa: E402


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
