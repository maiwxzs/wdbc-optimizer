"""Carga y preprocesamiento del conjunto WDBC.

Scikit-learn se usa EXCLUSIVAMENTE para: load_breast_cancer, train_test_split,
StandardScaler y PCA (2D, solo visualización). El entrenamiento es NumPy puro.

Convención de etiquetas: en sklearn target=0 es maligno y target=1 benigno.
Aquí se re-codifica y = 1 - target para que y = 1 signifique MALIGNO, de modo
que σ(z) se lea directamente como P(maligno | x).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_BASE_ES = {
    "radius": "Radio",
    "texture": "Textura",
    "perimeter": "Perímetro",
    "area": "Área",
    "smoothness": "Suavidad",
    "compactness": "Compacidad",
    "concavity": "Concavidad",
    "concave points": "Puntos cóncavos",
    "symmetry": "Simetría",
    "fractal dimension": "Dimensión fractal",
}
_STAT_ES = {"mean": "media", "error": "error est.", "worst": "peor"}


def translate_feature(name: str) -> str:
    for stat_en, stat_es in _STAT_ES.items():
        if name.startswith(stat_en + " "):
            base = name[len(stat_en) + 1 :]
            return f"{_BASE_ES.get(base, base)} ({stat_es})"
        if name.endswith(" " + stat_en):
            base = name[: -(len(stat_en) + 1)]
            return f"{_BASE_ES.get(base, base)} ({stat_es})"
    return name


@dataclass(frozen=True)
class WDBCDataset:
    feature_names: list[str]
    feature_names_es: list[str]
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    scaler: StandardScaler
    pca: PCA
    pca_train: np.ndarray
    pca_test: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def standardize(self, X_raw: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.atleast_2d(np.asarray(X_raw, dtype=np.float64)))

    def project(self, X_std: np.ndarray) -> np.ndarray:
        return self.pca.transform(np.atleast_2d(X_std))


@lru_cache(maxsize=4)
def load_wdbc(test_size: float = 0.2, random_state: int = 42) -> WDBCDataset:
    bunch = load_breast_cancer()
    X = bunch.data.astype(np.float64)
    y = (1 - bunch.target).astype(np.float64)  # 1 = maligno
    idx = np.arange(X.shape[0])

    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)  # ajustado solo con train: sin fuga de datos
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    pca = PCA(n_components=2, random_state=random_state).fit(Xs_tr)

    names = [str(n) for n in bunch.feature_names]
    return WDBCDataset(
        feature_names=names,
        feature_names_es=[translate_feature(n) for n in names],
        X_train_raw=X_tr,
        X_test_raw=X_te,
        X_train=Xs_tr,
        X_test=Xs_te,
        y_train=y_tr,
        y_test=y_te,
        idx_train=i_tr,
        idx_test=i_te,
        scaler=scaler,
        pca=pca,
        pca_train=pca.transform(Xs_tr),
        pca_test=pca.transform(Xs_te),
    )
