"""Online CE-inspired cluster-correlation linear occupancy surrogates.

The caller supplies current-lattice cluster-correlation features. Models train
only on energies observed in the current run and perform no filesystem access.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge

from agentic_policy_v2 import Candidate, Selection


class ClusterExpansionSurrogate:
    """Regularized cluster-correlation linear fit with posterior uncertainty."""

    SUPPORTED = ("bayesian_ridge", "ard")

    def __init__(self, kind: str = "bayesian_ridge"):
        if kind not in self.SUPPORTED:
            raise ValueError(f"unsupported cluster-expansion surrogate: {kind}")
        self.kind = kind
        self.offset_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.model = None

    def fit(self, features: np.ndarray, energies: np.ndarray):
        x = np.asarray(features, dtype=float)
        y = np.asarray(energies, dtype=float)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("features must be 2D and aligned with 1D energies")
        self.offset_ = x.mean(axis=0)
        # Correlations are bounded near [-1, 1]. A physical-scale floor avoids
        # exploding a feature that happened to be constant in early history.
        self.scale_ = np.maximum(x.std(axis=0), 0.25)
        scaled = (x - self.offset_) / self.scale_
        if self.kind == "bayesian_ridge":
            self.model = BayesianRidge(max_iter=500, tol=1e-6, compute_score=False)
        else:
            self.model = ARDRegression(max_iter=500, tol=1e-4, threshold_lambda=1e4)
        self.model.fit(scaled, y)
        return self

    def predict(self, features: np.ndarray, return_std: bool = False):
        if self.model is None or self.offset_ is None or self.scale_ is None:
            raise RuntimeError("cluster-expansion surrogate is not fitted")
        x = np.asarray(features, dtype=float)
        scaled = (x - self.offset_) / self.scale_
        if return_std:
            mean, std = self.model.predict(scaled, return_std=True)
            return np.asarray(mean, float), np.maximum(np.asarray(std, float), 0.0)
        return np.asarray(self.model.predict(scaled), float)


def _scaled_features(candidates: Sequence[Candidate], history_occ: Sequence[Sequence[str]],
                     feature_fn: Callable[[Sequence[str]], np.ndarray]):
    candidate_features = np.asarray([feature_fn(item.occupancy) for item in candidates], float)
    history_features = np.asarray([feature_fn(item) for item in history_occ], float)
    combined = np.vstack((history_features, candidate_features))
    scale = np.maximum(combined.std(axis=0), 0.25)
    return candidate_features / scale, history_features / scale


def acquisition_scores(mode: str, mean: np.ndarray, std: np.ndarray,
                       novelty: np.ndarray, energy_scale: float) -> np.ndarray:
    """Return scores whose semantics exactly match the credited mode."""
    normalized_novelty = novelty / max(float(novelty.max()), 1e-12)
    if mode == "greedy":
        return np.asarray(mean, float)
    if mode == "uncertainty":
        return -np.asarray(std, float)
    if mode == "lcb":
        return (np.asarray(mean, float) - 1.5 * np.asarray(std, float)
                - 0.08 * energy_scale * normalized_novelty)
    raise ValueError(f"unknown acquisition mode: {mode}")


def select_candidate_ce(candidates: Sequence[Candidate],
                        history_occ: Sequence[Sequence[str]],
                        history_energy: Sequence[float],
                        feature_fn: Callable[[Sequence[str]], np.ndarray],
                        kind: str, acquisition: str, seed: int = 0) -> Selection:
    """Select with an online linear cluster expansion and posterior uncertainty."""
    del seed  # linear Bayesian fits are deterministic
    if not candidates:
        raise ValueError("candidate list is empty")
    xc, xh = _scaled_features(candidates, history_occ, feature_fn)
    novelty = np.asarray([min(np.linalg.norm(x - y) for y in xh) for x in xc])
    if len(history_energy) < 8 or acquisition == "novelty":
        index = int(np.argmax(novelty))
        return Selection(index, None, None, float(novelty[index]), "descriptor novelty")

    model = ClusterExpansionSurrogate(kind).fit(xh, np.asarray(history_energy, float))
    mean, std = model.predict(xc, return_std=True)
    energy_scale = max(float(np.std(history_energy)), 1e-3)
    score = acquisition_scores(acquisition, mean, std, novelty, energy_scale)
    index = int(np.argmin(score))
    return Selection(index, float(mean[index]), float(std[index]),
                     float(novelty[index]),
                     f"online CE-inspired cluster correlation {kind} {acquisition}")
