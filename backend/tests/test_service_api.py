import json

import numpy as np
import pytest

from app.evaluation.metrics import roc_auc
from app.services.model_service import TrainRequestDTO


def _is_strict_json(obj) -> bool:
    json.dumps(obj, allow_nan=False)
    return True


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 80)
    s = np.round(rng.random(80), 1)  # con empates
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert roc_auc(y, s) == pytest.approx(brute)


def test_service_default_model(service):
    info = service.model_info()
    assert info["metrics"]["test"]["accuracy"] > 0.95
    assert _is_strict_json(info)


def test_service_train_divergent_is_json_safe(service):
    out = service.train(TrainRequestDTO(alpha=1e5, max_iter=50, mode="naive"))
    assert out["diverged"] and not out["active_updated"]
    assert _is_strict_json(out)


def test_service_predict_paths(service):
    by_idx = service.predict(sample_index=0)
    assert 0.0 <= by_idx["probability_malignant"] <= 1.0
    assert by_idx["true_label"] in (0, 1)
    by_feat = service.predict(features=by_idx["raw_features"])
    assert by_feat["probability_malignant"] == pytest.approx(by_idx["probability_malignant"], rel=1e-9)
    z = sum(c["contribution"] for c in by_idx["contributions"]) + by_idx["bias"]
    assert z == pytest.approx(by_idx["logit"])
    with pytest.raises(ValueError):
        service.predict(features=[1.0, 2.0])


def test_service_compare_and_audits(service):
    cmp = service.compare(max_iter=300)
    reasons = {r["label"]: r["stop_reason"] for r in cmp["runs"]}
    assert reasons["Divergente α=1e5"] == "divergence"
    assert reasons["Ingenuo α=1 (sin estabilizar)"] == "divergence"
    assert _is_strict_json(cmp)
    gc = service.gradient_check(point="random")
    assert gc["verdict"] == "excelente" and len(gc["components"]) == 31
    sw = service.step_sweep(n=20)
    assert len(sw["h"]) == 20 and _is_strict_json(sw)
    fa = service.float_audit()
    assert _is_strict_json(fa)


def test_fastapi_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/dataset/summary").json()["n_features"] == 30
        r = client.post("/api/train", json={"alpha": 2.0, "max_iter": 500})
        assert r.status_code == 200 and "history" in r.json()
        assert client.post("/api/predict", json={}).status_code == 422
        assert client.post("/api/predict", json={"sample_index": 3}).status_code == 200
        assert client.post("/api/audit/gradient-check", json={"point": "zeros"}).status_code == 200
        assert client.get("/api/audit/float").status_code == 200
