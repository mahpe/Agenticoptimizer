from pathlib import Path
import inspect
import sys

import numpy as np
import pytest
from ase import Atoms

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from agentic_policy_aliegnn_online import (  # noqa: E402
    CompactALiEGNN,
    FixedLineGraphGeometry,
    OnlineALiEGNNEnsemble,
    select_candidate_aliegnn,
)
from agentic_policy_v2 import Candidate  # noqa: E402


def geometry():
    atoms = Atoms(
        symbols=['Sb'] * 8,
        scaled_positions=[
            (0, 0, 0), (.5, 0, 0), (0, .5, 0), (0, 0, .5),
            (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.5, .5, .5),
        ],
        cell=[8, 8, 8], pbc=True,
    )
    return FixedLineGraphGeometry.from_atoms(atoms, cutoff=7.0, max_neighbors=6)


def occupancies(n=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        row = ['Sb'] * 4 + ['Te'] * 4
        rng.shuffle(row)
        if row not in rows:
            rows.append(row.copy())
    return rows


def energy(occ):
    code = np.asarray([1.0 if x == 'Te' else -1.0 for x in occ])
    geom = geometry()
    pair = code[geom.edge_sources] * code[geom.edge_targets]
    return float(0.5 * np.sum(pair * np.exp(-geom.distances / 4.0)))


def feature(occ):
    code = np.asarray([1.0 if x == 'Te' else -1.0 for x in occ])
    return np.concatenate([code, code * np.roll(code, 1)])


def candidates(rows):
    return [Candidate(row.copy(), [], 0, tuple(row)) for row in rows]


def test_geometry_builds_bounded_atom_and_line_graphs():
    geom = geometry()
    assert geom.num_nodes == 8
    assert 0 < geom.num_edges <= 8 * 6
    assert geom.num_line_edges > 0
    assert np.all(geom.distances > 0)
    assert np.all((-1.0 <= geom.angle_cosines) & (geom.angle_cosines <= 1.0))


def test_geometry_is_invariant_to_periodic_translation():
    atoms = Atoms(
        symbols=['Sb'] * 8,
        scaled_positions=[
            (0, 0, 0), (.5, 0, 0), (0, .5, 0), (0, 0, .5),
            (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.5, .5, .5),
        ], cell=[8, 8, 8], pbc=True,
    )
    original = FixedLineGraphGeometry.from_atoms(atoms, cutoff=7.0, max_neighbors=6)
    atoms.translate([3.1, -2.4, 1.7])
    translated = FixedLineGraphGeometry.from_atoms(atoms, cutoff=7.0, max_neighbors=6)
    np.testing.assert_array_equal(original.edge_sources, translated.edge_sources)
    np.testing.assert_array_equal(original.edge_targets, translated.edge_targets)
    np.testing.assert_allclose(original.distances, translated.distances, atol=1e-6)
    np.testing.assert_allclose(original.angle_cosines, translated.angle_cosines, atol=1e-6)


def test_graph_resource_limits_fail_before_materialization():
    too_many = Atoms('H257', positions=np.zeros((257, 3)), cell=[20, 20, 20], pbc=True)
    with pytest.raises(ValueError, match='graph node'):
        FixedLineGraphGeometry.from_atoms(too_many)
    with pytest.raises(ValueError, match='line-graph bound'):
        FixedLineGraphGeometry.from_atoms(
            Atoms('H8', positions=np.arange(24).reshape(8, 3), cell=[30, 30, 30], pbc=True),
            max_neighbors=10_000,
        )


def test_compact_model_is_small_and_returns_one_energy_per_occupancy():
    geom = geometry()
    model = CompactALiEGNN(geom, num_species=2, hidden_dim=24, layers=2)
    count = sum(p.numel() for p in model.parameters())
    assert count < 100_000
    values = model.encode_occupancies(occupancies(5), ('Sb', 'Te'))
    prediction = model(values)
    assert tuple(prediction.shape) == (5,)


def test_online_ensemble_learns_synthetic_occupancy_energy():
    rows = occupancies(24)
    train, test = rows[:18], rows[18:]
    surrogate = OnlineALiEGNNEnsemble(
        geometry(), ('Sb', 'Te'), ensemble_size=3, hidden_dim=24,
        layers=2, epochs=100, learning_rate=8e-3, seed=7,
    )
    surrogate.fit(train, np.asarray([energy(x) for x in train]))
    mean, std = surrogate.predict(test, return_std=True)
    truth = np.asarray([energy(x) for x in test])
    baseline = np.mean((truth - np.mean([energy(x) for x in train])) ** 2)
    mse = np.mean((truth - mean) ** 2)
    assert mse < baseline
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std)) and np.all(std >= 0)


def test_selection_is_deterministic_and_does_not_mutate_candidates():
    rows = occupancies(24)
    history, pool = rows[:12], rows[12:]
    items = candidates(pool)
    before = [x.occupancy.copy() for x in items]
    kwargs = dict(
        candidates=items, history_occ=history,
        history_energy=[energy(x) for x in history], geometry=geometry(),
        species=('Sb', 'Te'), feature_fn=feature, acquisition='lcb', seed=3,
        ensemble_size=3, hidden_dim=16, layers=1, epochs=40,
    )
    first = select_candidate_aliegnn(**kwargs)
    second = select_candidate_aliegnn(**kwargs)
    assert first == second
    assert 0 <= first.index < len(items)
    assert first.predicted_energy is not None
    assert first.uncertainty is not None and first.uncertainty >= 0
    assert len(first.training_data_sha256) == 64
    assert len(first.ensemble_state_sha256) == 64
    assert first.training_data_sha256 == second.training_data_sha256
    assert first.ensemble_state_sha256 == second.ensemble_state_sha256
    assert [x.occupancy for x in items] == before


def test_training_digest_changes_when_an_observed_label_changes():
    rows = occupancies(20)
    items = candidates(rows[12:])
    energies = [energy(x) for x in rows[:12]]
    common = dict(candidates=items, history_occ=rows[:12], geometry=geometry(),
                  species=('Sb', 'Te'), feature_fn=feature, acquisition='lcb', seed=5,
                  ensemble_size=2, hidden_dim=12, layers=1, epochs=5)
    first = select_candidate_aliegnn(history_energy=energies, **common)
    changed = energies.copy(); changed[-1] += 0.25
    second = select_candidate_aliegnn(history_energy=changed, **common)
    assert first.training_data_sha256 != second.training_data_sha256
    assert first.ensemble_state_sha256 != second.ensemble_state_sha256


def test_candidate_and_history_resource_limits_fail_closed():
    rows = occupancies(24)
    item = candidates(rows[:1])[0]
    with pytest.raises(ValueError, match='candidate batch'):
        select_candidate_aliegnn(
            [item] * 3001, rows[:8], [energy(x) for x in rows[:8]],
            geometry(), ('Sb', 'Te'), feature, acquisition='lcb', seed=0)
    with pytest.raises(ValueError, match='training history'):
        select_candidate_aliegnn(
            [item], [rows[0]] * 202, [energy(rows[0])] * 202,
            geometry(), ('Sb', 'Te'), feature, acquisition='lcb', seed=0)


def test_warmup_matches_existing_eight_observation_contract():
    rows = occupancies(16)
    selection = select_candidate_aliegnn(
        candidates(rows[7:]), rows[:7], [energy(x) for x in rows[:7]],
        geometry(), ('Sb', 'Te'), feature, acquisition='lcb', seed=0,
    )
    assert selection.predicted_energy is None
    assert selection.uncertainty is None
    assert selection.selector == 'descriptor novelty'


def test_production_defaults_are_bounded_for_online_retraining():
    parameters = inspect.signature(select_candidate_aliegnn).parameters
    assert parameters['ensemble_size'].default == 3
    assert parameters['hidden_dim'].default == 16
    assert parameters['layers'].default == 1
    assert parameters['epochs'].default == 20


def test_module_has_no_filesystem_or_external_data_access():
    source = (ROOT/'agentic_policy_aliegnn_online.py').read_text()
    forbidden = ['open(', 'read_text(', 'read_bytes(', 'torch.load(', 'enumeration', 'reference_energy']
    assert not any(term in source for term in forbidden)
