import numpy as np
import pytest

from agentic_policy_v2 import (
    balanced_jump_neighbors,
    diverse_elite_archive,
    effective_acquisition,
    elite_stop_evidence,
    hamming_distance,
    jump_space_exhausted,
    select_candidate_v2,
    single_neighbors_from_parent,
    tree_acquisition_scores,
)


def test_balanced_four_swap_jump_preserves_exact_composition_and_is_unseen():
    occupancy = ["A"] * 8 + ["B"] * 8
    seen = {tuple(occupancy)}
    candidates = balanced_jump_neighbors(occupancy, seen, 4, np.random.default_rng(3), limit=100)
    assert candidates
    for candidate in candidates:
        assert candidate.num_swaps == 4
        assert candidate.occupancy.count("A") == 8
        assert candidate.occupancy.count("B") == 8
        assert tuple(candidate.occupancy) not in seen
        assert hamming_distance(candidate.occupancy, occupancy) == 8


def test_diverse_elite_archive_keeps_best_and_separate_run_local_basins():
    occupancies = [
        ["A", "A", "A", "A", "B", "B", "B", "B"],
        ["A", "B", "A", "A", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "A", "A", "B", "B"],
        ["A", "B", "A", "B", "A", "B", "A", "B"],
        ["B", "A", "A", "A", "A", "B", "B", "B"],
    ]
    energies = [-10.0, -9.99, -9.98, -9.97, -9.96]
    elites = diverse_elite_archive(occupancies, energies, max_elites=3, min_hamming=4)
    assert elites[0].energy == -10.0
    assert len(elites) == 3
    assert all(hamming_distance(a.occupancy, b.occupancy) >= 4
               for i, a in enumerate(elites) for b in elites[i + 1 :])


def test_single_neighbors_are_generated_from_each_elite_not_only_global_best():
    parent_a = ["A", "A", "B", "B"]
    parent_b = ["A", "B", "A", "B"]
    distances = np.ones((4, 4)) - np.eye(4)
    seen = {tuple(parent_a)}
    a = single_neighbors_from_parent(parent_a, seen, distances)
    b = single_neighbors_from_parent(parent_b, seen, distances)
    assert a and b
    assert {c.parent_fingerprint for c in a} == {tuple(parent_a)}
    assert {c.parent_fingerprint for c in b} == {tuple(parent_b)}


def test_model_ranked_jump_returns_predictions_instead_of_forced_novelty():
    history = [
        ["A", "A", "B", "B"], ["A", "B", "A", "B"],
        ["A", "B", "B", "A"], ["B", "A", "A", "B"],
        ["B", "A", "B", "A"], ["B", "B", "A", "A"],
        ["A", "A", "B", "B"], ["B", "B", "A", "A"],
    ]
    energies = [-4.0, -3.0, -2.0, -3.1, -2.1, -1.0, -4.0, -1.0]
    candidates = balanced_jump_neighbors(history[0], {tuple(history[0])}, 1,
                                         np.random.default_rng(1), limit=20)
    feature = lambda occ: np.asarray([1.0 if x == "B" else 0.0 for x in occ])
    result = select_candidate_v2(candidates, history, energies, feature, seed=0,
                                 acquisition_mode="lcb")
    assert result.predicted_energy is not None
    assert result.uncertainty is not None
    assert result.selector == "online ExtraTrees lcb"


def test_warmup_records_the_acquisition_actually_executed():
    assert effective_acquisition("lcb", 7) == "novelty"
    assert effective_acquisition("uncertainty", 8) == "uncertainty"
    assert effective_acquisition("novelty", 2) == "novelty"


def test_tree_greedy_and_uncertainty_have_exact_matched_semantics():
    mean = np.asarray([2.0, 1.0, 3.0])
    std = np.asarray([0.1, 0.2, 0.9])
    novelty = np.asarray([100.0, 0.0, 1.0])
    assert np.array_equal(tree_acquisition_scores("greedy", mean, std, novelty, 1.0), mean)
    assert np.array_equal(tree_acquisition_scores("uncertainty", mean, std, novelty, 1.0), -std)


def test_large_jump_space_is_rejected_before_unbounded_materialization():
    occupancy = ["A"] * 30 + ["B"] * 30
    with pytest.raises(ValueError, match="supported exhaustive limit"):
        balanced_jump_neighbors(occupancy, {tuple(occupancy)}, 4,
                                np.random.default_rng(0), limit=20)


def test_jump_sampling_filters_seen_before_applying_limit():
    occupancy = ["A"] * 4 + ["B"] * 4
    all_candidates = balanced_jump_neighbors(occupancy, {tuple(occupancy)}, 2,
                                             np.random.default_rng(0), limit=100)
    target = all_candidates[-1]
    seen = {tuple(occupancy)} | {tuple(c.occupancy) for c in all_candidates[:-1]}
    candidates = balanced_jump_neighbors(occupancy, seen, 2,
                                         np.random.default_rng(0), limit=1)
    assert [tuple(c.occupancy) for c in candidates] == [tuple(target.occupancy)]
    assert jump_space_exhausted(occupancy, seen, 2) is False
    assert jump_space_exhausted(occupancy, seen | {tuple(target.occupancy)}, 2) is True


def test_stop_requires_all_elite_neighborhoods_and_each_acquisition_mode():
    elite_fps = [("A", "A", "B", "B"), ("A", "B", "A", "B")]
    modes = ("lcb", "greedy", "uncertainty", "novelty")
    mode_counts = {(fp, size, mode): 1 for fp in elite_fps
                   for size in (2, 3, 4) for mode in modes}
    evidence = elite_stop_evidence(
        step=100,
        proposals_since_improvement=50,
        elite_fingerprints=elite_fps,
        unseen_single_counts={elite_fps[0]: 0, elite_fps[1]: 0},
        jump_mode_counts=mode_counts,
        exhausted_jump_slots=set(),
        jump_sizes=(2, 3, 4), acquisition_modes=modes, stable_elite_steps=30,
    )
    assert evidence["done"] is True
    mode_counts[(elite_fps[1], 4, "novelty")] = 0
    evidence2 = elite_stop_evidence(100, 50, elite_fps,
                                    {elite_fps[0]: 0, elite_fps[1]: 0},
                                    mode_counts, set(), (2, 3, 4), modes, 30)
    assert evidence2["done"] is False
    assert evidence2["all_elite_jump_modes_met_or_exhausted"] is False
    evidence3 = elite_stop_evidence(100, 50, elite_fps,
                                    {elite_fps[0]: 0, elite_fps[1]: 0},
                                    mode_counts, {(elite_fps[1], 4)},
                                    (2, 3, 4), modes, 30)
    assert evidence3["done"] is True
