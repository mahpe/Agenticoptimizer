import numpy as np
import pytest

from agentic_policy_v2 import Candidate
from agentic_policy_ce import (
    ClusterExpansionSurrogate,
    acquisition_scores,
    select_candidate_ce,
)


@pytest.mark.parametrize("kind", ["bayesian_ridge", "ard"])
def test_cluster_expansion_surrogate_returns_finite_mean_and_uncertainty(kind):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 7))
    coefficients = np.array([1.2, 0.0, -0.7, 0.0, 0.3, 0.0, 0.5])
    y = -12.0 + x @ coefficients
    model = ClusterExpansionSurrogate(kind=kind).fit(x, y)
    mean, std = model.predict(x[:6], return_std=True)
    assert mean.shape == (6,)
    assert std.shape == (6,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)
    assert np.max(np.abs(mean - y[:6])) < 0.1


def test_cluster_expansion_surrogate_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported cluster-expansion surrogate"):
        ClusterExpansionSurrogate(kind="tree")


def test_uncertainty_mode_selects_maximum_std_with_novelty_only_as_tiebreak():
    mean = np.array([-10.0, -9.0, -8.0])
    std = np.array([0.2, 0.5, 0.3])
    novelty = np.array([100.0, 0.0, 50.0])
    score = acquisition_scores("uncertainty", mean, std, novelty, 1.0)
    assert int(np.argmin(score)) == 1


def test_greedy_mode_selects_lowest_mean_with_novelty_only_as_tiebreak():
    mean = np.array([-10.0, -9.0, -8.0])
    std = np.array([0.2, 100.0, 100.0])
    novelty = np.array([0.0, 100.0, 100.0])
    score = acquisition_scores("greedy", mean, std, novelty, 1.0)
    assert int(np.argmin(score)) == 0


def test_correlation_feature_scaling_does_not_amplify_constant_columns():
    x = np.column_stack((np.ones(12), np.linspace(-1, 1, 12)))
    y = np.linspace(-2, -1, 12)
    model = ClusterExpansionSurrogate("bayesian_ridge").fit(x, y)
    assert model.scale_[0] >= 0.25


def test_ce_selector_uses_cluster_model_after_warmup():
    history = [
        ["A", "A", "B", "B"], ["A", "B", "A", "B"],
        ["A", "B", "B", "A"], ["B", "A", "A", "B"],
        ["B", "A", "B", "A"], ["B", "B", "A", "A"],
        ["A", "A", "B", "B"], ["B", "B", "A", "A"],
    ]
    energies = [-4.0, -3.4, -3.1, -3.3, -3.0, -3.8, -4.01, -3.79]
    candidates = [
        Candidate(["A", "B", "A", "B"], [[1, 2]], 1, tuple(history[0])),
        Candidate(["B", "A", "B", "A"], [[0, 2]], 1, tuple(history[0])),
    ]

    def features(occupancy):
        spins = np.array([1.0 if value == "A" else -1.0 for value in occupancy])
        return np.concatenate([spins, [spins[0]*spins[1], spins[2]*spins[3]]])

    result = select_candidate_ce(
        candidates, history, energies, features, kind="bayesian_ridge",
        acquisition="lcb", seed=3,
    )
    assert result.index in (0, 1)
    assert result.predicted_energy is not None
    assert result.uncertainty is not None
    assert result.selector == "online CE-inspired cluster correlation bayesian_ridge lcb"
