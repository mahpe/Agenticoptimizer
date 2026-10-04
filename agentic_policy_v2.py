"""Reusable strict-isolation occupancy-search policy primitives.

All functions operate only on the current system and current-run observations.
They contain no filesystem access and no reference-energy knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


Fingerprint = tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    occupancy: list[str]
    swaps: list[list[int]]
    num_swaps: int
    parent_fingerprint: Fingerprint
    pair_distance: float = float("nan")


@dataclass(frozen=True)
class Elite:
    occupancy: list[str]
    energy: float
    history_index: int

    @property
    def fingerprint(self) -> Fingerprint:
        return tuple(self.occupancy)


@dataclass(frozen=True)
class Selection:
    index: int
    predicted_energy: float | None
    uncertainty: float | None
    novelty: float
    selector: str


def hamming_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) != len(right):
        raise ValueError("occupancies must have the same length")
    return sum(a != b for a, b in zip(left, right))


MAX_EXHAUSTIVE_JUMP_SPACE = 200_000


def effective_acquisition(requested: str, history_size: int) -> str:
    """Return the acquisition the selector can actually execute."""
    return "novelty" if history_size < 8 else requested


def _jump_basis(parent: Sequence[str], k: int):
    parent = list(parent)
    species = list(dict.fromkeys(parent))
    if len(species) != 2 or k < 1:
        return parent, [], [], 0, 0, 0
    left = [i for i, value in enumerate(parent) if value == species[0]]
    right = [i for i, value in enumerate(parent) if value == species[1]]
    if k > min(len(left), len(right)):
        return parent, [], [], 0, 0, 0
    left_count = math.comb(len(left), k)
    right_count = math.comb(len(right), k)
    return parent, left, right, left_count, right_count, left_count * right_count


def _unrank_combination(items: Sequence[int], k: int, rank: int) -> tuple[int, ...]:
    """Lexicographically unrank one k-combination without listing all combinations."""
    n = len(items)
    if rank < 0 or rank >= math.comb(n, k):
        raise IndexError(rank)
    result: list[int] = []
    start = 0
    remaining = k
    while remaining:
        for index in range(start, n - remaining + 1):
            block = math.comb(n - index - 1, remaining - 1)
            if rank < block:
                result.append(items[index])
                start = index + 1
                remaining -= 1
                break
            rank -= block
    return tuple(result)


def _jump_at(parent: list[str], left: Sequence[int], right: Sequence[int],
             left_count: int, right_count: int, flat_index: int, k: int) -> Candidate:
    del left_count  # retained in the signature for explicit basis provenance
    aa = _unrank_combination(left, k, flat_index // right_count)
    bb = _unrank_combination(right, k, flat_index % right_count)
    trial = parent.copy()
    swaps: list[list[int]] = []
    for i, j in zip(aa, bb):
        trial[i], trial[j] = trial[j], trial[i]
        swaps.append([i, j])
    return Candidate(trial, swaps, k, tuple(parent))


def jump_space_size(parent: Sequence[str], k: int) -> int:
    """Return the exact number of binary balanced k-pair exchanges."""
    return int(_jump_basis(parent, k)[-1])


def balanced_jump_neighbors(parent: Sequence[str], seen: set[Fingerprint], k: int,
                            rng: np.random.Generator, limit: int = 3000) -> list[Candidate]:
    """Return unseen k-pair exchanges from a fully enumerable supported space."""
    parent, left, right, left_count, right_count, total = _jump_basis(parent, k)
    if total == 0:
        return []
    if total > MAX_EXHAUSTIVE_JUMP_SPACE:
        raise ValueError(
            f"jump space {total} exceeds supported exhaustive limit "
            f"{MAX_EXHAUSTIVE_JUMP_SPACE}; reject this configuration before evaluation")
    unseen = [_jump_at(parent, left, right, left_count, right_count, index, k)
              for index in range(total)]
    unseen = [candidate for candidate in unseen if tuple(candidate.occupancy) not in seen]
    if len(unseen) <= limit:
        return unseen
    chosen = sorted(int(i) for i in rng.choice(len(unseen), size=limit, replace=False))
    return [unseen[i] for i in chosen]


def jump_space_exhausted(parent: Sequence[str], seen: set[Fingerprint], k: int) -> bool:
    """Prove exhaustion within the explicitly supported finite jump space."""
    parent, left, right, left_count, right_count, total = _jump_basis(parent, k)
    if total == 0:
        return True
    if total > MAX_EXHAUSTIVE_JUMP_SPACE:
        raise ValueError(
            f"jump space {total} exceeds supported exhaustive limit "
            f"{MAX_EXHAUSTIVE_JUMP_SPACE}")
    return all(tuple(_jump_at(parent, left, right, left_count, right_count, index, k).occupancy) in seen
               for index in range(total))


def single_neighbors_from_parent(parent: Sequence[str], seen: set[Fingerprint],
                                 distances: np.ndarray) -> list[Candidate]:
    parent = list(parent)
    result: list[Candidate] = []
    parent_fp = tuple(parent)
    for i in range(len(parent)):
        for j in range(i + 1, len(parent)):
            if parent[i] == parent[j]:
                continue
            trial = parent.copy()
            trial[i], trial[j] = trial[j], trial[i]
            if tuple(trial) in seen:
                continue
            result.append(Candidate(trial, [[i, j]], 1, parent_fp,
                                    float(distances[i, j])))
    return result


def diverse_elite_archive(occupancies: Sequence[Sequence[str]], energies: Sequence[float],
                          max_elites: int = 4, min_hamming: int = 4) -> list[Elite]:
    """Select low-energy, mutually separated basins from current-run history."""
    if len(occupancies) != len(energies):
        raise ValueError("occupancy and energy history lengths differ")
    order = sorted(range(len(energies)), key=lambda i: (float(energies[i]), i))
    elites: list[Elite] = []
    seen: set[Fingerprint] = set()
    for index in order:
        occupancy = list(occupancies[index])
        fp = tuple(occupancy)
        if fp in seen:
            continue
        if elites and any(hamming_distance(occupancy, elite.occupancy) < min_hamming
                          for elite in elites):
            continue
        elites.append(Elite(occupancy, float(energies[index]), index))
        seen.add(fp)
        if len(elites) == max_elites:
            break
    return elites


def _scaled_features(candidates: Sequence[Candidate], history_occ: Sequence[Sequence[str]],
                     feature_fn: Callable[[Sequence[str]], np.ndarray]):
    xc = np.asarray([feature_fn(item.occupancy) for item in candidates], dtype=float)
    xh = np.asarray([feature_fn(item) for item in history_occ], dtype=float)
    combined = np.vstack((xh, xc))
    scale = np.maximum(combined.std(axis=0), 0.25)
    return xc / scale, xh / scale


def tree_acquisition_scores(mode: str, mean: np.ndarray, std: np.ndarray,
                            novelty: np.ndarray, energy_scale: float) -> np.ndarray:
    """Matched acquisition semantics for the ExtraTrees comparison arm."""
    normalized_novelty = novelty / max(float(novelty.max()), 1e-12)
    if mode == "greedy":
        return np.asarray(mean, float)
    if mode == "uncertainty":
        return -np.asarray(std, float)
    if mode == "lcb":
        return (np.asarray(mean, float) - 1.5 * np.asarray(std, float)
                - 0.08 * energy_scale * normalized_novelty)
    raise ValueError(f"unknown acquisition mode: {mode}")


def select_candidate_v2(candidates: Sequence[Candidate], history_occ: Sequence[Sequence[str]],
                        history_energy: Sequence[float],
                        feature_fn: Callable[[Sequence[str]], np.ndarray], seed: int,
                        acquisition_mode: str = "lcb") -> Selection:
    if not candidates:
        raise ValueError("candidate list is empty")
    xc, xh = _scaled_features(candidates, history_occ, feature_fn)
    novelty = np.asarray([min(np.linalg.norm(x - y) for y in xh) for x in xc])
    if len(history_energy) < 8 or acquisition_mode == "novelty":
        index = int(np.argmax(novelty))
        return Selection(index, None, None, float(novelty[index]), "descriptor novelty")
    model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.9,
                                random_state=seed, n_jobs=-1)
    model.fit(xh, np.asarray(history_energy, dtype=float))
    ensemble = np.asarray([tree.predict(xc) for tree in model.estimators_])
    mean, std = ensemble.mean(axis=0), ensemble.std(axis=0)
    energy_scale = max(float(np.std(history_energy)), 1e-3)
    acquisition = tree_acquisition_scores(
        acquisition_mode, mean, std, novelty, energy_scale)
    index = int(np.argmin(acquisition))
    return Selection(index, float(mean[index]), float(std[index]),
                     float(novelty[index]), f"online ExtraTrees {acquisition_mode}")


def elite_stop_evidence(step: int, proposals_since_improvement: int,
                       elite_fingerprints: Sequence[Fingerprint],
                       unseen_single_counts: dict[Fingerprint, int],
                       jump_mode_counts: dict[tuple[Fingerprint, int, str], int],
                       exhausted_jump_slots: set[tuple[Fingerprint, int]],
                       jump_sizes: Sequence[int], acquisition_modes: Sequence[str],
                       stable_elite_steps: int) -> dict:
    all_single = bool(elite_fingerprints) and all(
        unseen_single_counts.get(fp, 1) == 0 for fp in elite_fingerprints)
    all_jump_modes = bool(elite_fingerprints) and all(
        (fp, size) in exhausted_jump_slots or
        all(jump_mode_counts.get((fp, size, mode), 0) >= 1 for mode in acquisition_modes)
        for fp in elite_fingerprints for size in jump_sizes)
    evidence = {
        "minimum_proposals_met": step >= 32,
        "elite_count": len(elite_fingerprints),
        "all_elite_single_neighborhoods_evaluated": all_single,
        "all_elite_jump_modes_met_or_exhausted": all_jump_modes,
        "jump_sizes": list(jump_sizes),
        "required_acquisition_modes": list(acquisition_modes),
        "exhausted_jump_slots": [
            {"parent_fingerprint": list(fp), "size": size}
            for fp, size in sorted(exhausted_jump_slots)
        ],
        "proposals_since_last_improvement": proposals_since_improvement,
        "no_improvement_for_40_proposals": proposals_since_improvement >= 40,
        "stable_elite_steps": stable_elite_steps,
        "elite_archive_stable_for_25_proposals": stable_elite_steps >= 25,
    }
    required = ["minimum_proposals_met", "all_elite_single_neighborhoods_evaluated",
                "all_elite_jump_modes_met_or_exhausted", "no_improvement_for_40_proposals",
                "elite_archive_stable_for_25_proposals"]
    evidence["done"] = all(bool(evidence[key]) for key in required)
    return evidence
