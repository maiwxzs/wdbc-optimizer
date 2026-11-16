"""Prueba de humo del script de reproducción (sin figuras, en un directorio temporal)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reproduce_experiments.py"


def test_reproduce_experiments_writes_strict_json(tmp_path):
    spec = importlib.util.spec_from_file_location("reproduce_experiments", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--output", str(tmp_path), "--no-figures"])
    files = sorted(p.name for p in (tmp_path / "metrics").glob("*.json"))
    assert len(files) == 7
    e1 = json.loads((tmp_path / "metrics" / "e1_default_model.json").read_text(encoding="utf-8"))
    assert e1["stop_reason"] == "grad_norm" and e1["iterations"] == 4183
    assert e1["metrics_test"]["accuracy"] > 0.95
