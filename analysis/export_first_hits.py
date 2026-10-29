#!/usr/bin/env python3
"""Post-run, read-only first-hit export from explicitly supplied run/reference roots.

This script is NOT imported by the optimizer and cannot influence online selection.
It reports GA hit generations with evaluation intervals, never an invented exact hit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ase.io import read

SYSTEMS = (
    "AlReGe", "Ba2ScTaO6", "CoPt", "Cu3Au", "Cu3Au_123", "FeCo", "FeCo_331",
    "FePt", "InCuTe2", "LiYS2", "Ni2SbTe2", "ThMn2Co3", "TiCuNi", "V2ZnO4",
)
METHODS = ("random", "mc", "ga", "extra_trees", "bayesian_ridge", "ard", "aliegnn_online")
SURROGATES = set(METHODS[3:])
DEFAULT_PREFIX = "agentic_without_unique_database_all14_four_surrogates_five_seeds_v1_"
FIELDS = (
    "system", "method", "seed", "step_unit", "first_reference_hit_step",
    "first_hit_generation_eval_min", "first_hit_generation_eval_max",
    "energy_at_first_hit_eV", "best_energy_eV", "finite_reference_energy_eV",
    "best_gap_meV", "reference_hit", "hit_tolerance_eV", "last_recorded_step",
    "recorded_steps", "reported_relaxations", "source_summary",
)


def reference_energy(reference_root: Path, system: str) -> float:
    path = reference_root / system / "optimized_all.extxyz"
    frames = read(str(path), index=":")
    if not frames:
        raise ValueError(f"empty reference set: {path}")
    energies = [float(a.info["energy"]) if "energy" in a.info
                else float(a.get_potential_energy()) for a in frames]
    if not all(math.isfinite(value) for value in energies):
        raise ValueError(f"nonfinite reference energy: {path}")
    return min(energies)


def build_rows(*, traditional_root: Path, surrogate_root: Path, reference_root: Path,
               systems=SYSTEMS, methods=METHODS, seeds=range(5),
               tolerance_ev: float = 2e-5, surrogate_prefix: str = DEFAULT_PREFIX,
               expected_runner_sha256: str | None = None) -> list[dict]:
    """Require every requested summary; do not substitute missing arms with archive data."""
    if not math.isfinite(tolerance_ev) or tolerance_ev < 0:
        raise ValueError("tolerance_ev must be finite and nonnegative")
    if not set(methods) <= set(METHODS):
        raise ValueError(f"unknown methods: {set(methods) - set(METHODS)}")
    rows = []
    for system in systems:
        reference = reference_energy(Path(reference_root), system)
        for method in methods:
            for seed in seeds:
                folder = (f"{surrogate_prefix}{method}_seed{seed}" if method in SURROGATES
                          else f"{method}_seed{seed}")
                path = (Path(surrogate_root) if method in SURROGATES else Path(traditional_root)) / system / folder / "summary.json"
                data = json.loads(path.read_text())
                steps = data["steps"]
                indexes = [int(s["step"]) for s in steps]
                if not indexes or indexes[0] != 0 or any(a >= b for a, b in zip(indexes, indexes[1:])):
                    raise ValueError(f"invalid step sequence: {path}")
                if method != "random" and indexes != list(range(len(indexes))):
                    raise ValueError(f"noncontiguous {method} steps: {path}")
                if method in SURROGATES:
                    if (data.get("system") != system or data.get("surrogate_kind") != method
                            or data.get("seed") != seed or not data.get("stop_decision", {}).get("done")):
                        raise ValueError(f"unmatched/incomplete surrogate run: {path}")
                    if expected_runner_sha256 and data.get("runner_sha256") != expected_runner_sha256:
                        raise ValueError(f"runner digest mismatch: {path}")
                elif method == "ga" and (data.get("random_seed") != seed or
                                         int(data.get("population_size", 0)) <= 0):
                    raise ValueError(f"unmatched GA run: {path}")
                if method == "ga":
                    energies = [float(s["best_energy_eV"]) for s in steps]
                    if any(a < b for a, b in zip(energies, energies[1:])):
                        raise ValueError(f"GA best energy increased: {path}")
                    unit = "generation"
                    relaxations = int(data["relaxations"])
                    if relaxations < int(data["population_size"]) + len(indexes) - 1:
                        raise ValueError(f"GA relaxation count below recorded generations: {path}")
                else:
                    energies = [float(s["energy"]) for s in steps]
                    unit = "MC iteration" if method == "mc" else "proposal step"
                    relaxations = ""
                if not all(math.isfinite(e) for e in energies):
                    raise ValueError(f"nonfinite energy: {path}")
                best = min(energies)
                if not math.isclose(best, float(data["best_energy_eV"]), rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError(f"best energy mismatch: {path}")
                hit = next((i for i, energy in enumerate(energies)
                            if energy <= reference + tolerance_ev), None)
                generation = indexes[hit] if hit is not None else None
                pop = int(data["population_size"]) if method == "ga" else 0
                rows.append(dict(zip(FIELDS, (
                    system, method, seed, unit, "" if generation is None else generation,
                    (pop + generation if generation else 1) if generation is not None and pop else "",
                    min((generation + 1) * pop, relaxations) if generation is not None and pop else "",
                    energies[hit] if hit is not None else "", best, reference,
                    1000 * (best - reference), hit is not None, tolerance_ev,
                    indexes[-1], len(indexes), relaxations, str(path),
                ))))
    if len({(row["system"], row["method"], row["seed"]) for row in rows}) != len(rows):
        raise ValueError("duplicate (system, method, seed) requested")
    return rows


def write_rows(output: Path, rows: list[dict], *, run_roots: tuple[Path, ...]) -> None:
    """Create a semicolon CSV atomically, outside run roots, without overwriting."""
    output = Path(output)
    resolved_output = output.resolve()
    for root in run_roots:
        resolved_root = Path(root).resolve()
        if resolved_output == resolved_root or resolved_root in resolved_output.parents:
            raise ValueError(f"output must be outside run root: {root}")
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    with NamedTemporaryFile("w", newline="", encoding="utf-8", dir=output.parent,
                            prefix=".first_hits_", suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        # Hard-link creation is atomic and refuses an existing destination, even
        # if another exporter creates it between the first check and this call.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    with output.open(newline="", encoding="utf-8") as handle:
        verified = list(csv.DictReader(handle, delimiter=";"))
    if len(verified) != len(rows):
        raise RuntimeError("CSV read-back row count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("traditional-root", "surrogate-root", "reference-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tolerance-ev", type=float, default=2e-5)
    parser.add_argument("--surrogate-prefix", default=DEFAULT_PREFIX,
                        help="Override for archived all15-prefixed surrogate runs")
    parser.add_argument("--runner-source", type=Path,
                        help="Require each surrogate summary to identify this exact runner SHA-256")
    args = parser.parse_args()
    runner_hash = hashlib.sha256(args.runner_source.read_bytes()).hexdigest() if args.runner_source else None
    rows = build_rows(traditional_root=args.traditional_root, surrogate_root=args.surrogate_root,
                      reference_root=args.reference_root, tolerance_ev=args.tolerance_ev,
                      surrogate_prefix=args.surrogate_prefix, expected_runner_sha256=runner_hash)
    write_rows(args.output, rows, run_roots=(args.traditional_root, args.surrogate_root))
    print(f"Wrote {len(rows)} rows to {args.output}; tolerance={args.tolerance_ev:g} eV")


if __name__ == "__main__":
    main()
