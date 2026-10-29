"""Read-only post-run export: synthetic trajectories, no calculator or archived outcomes."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

MODULE = Path(__file__).resolve().parents[1] / "analysis" / "export_first_hits.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_first_hits", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_summary(root, folder, data):
    path = root / "Example" / folder / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_first_hits_use_recorded_steps_and_ga_generation_interval(tmp_path):
    exporter = load_exporter()
    traditional = tmp_path / "traditional"
    surrogate = tmp_path / "surrogates"
    reference = tmp_path / "references" / "Example"
    reference.mkdir(parents=True)
    atoms = Atoms("H", positions=[(0, 0, 0)], cell=[5, 5, 5], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0, forces=np.zeros((1, 3)))
    write(reference / "optimized_all.extxyz", atoms)
    make_summary(traditional, "random_seed0", {
        "best_energy_eV": -2.000001,
        "steps": [{"step": 0, "energy": -1}, {"step": 2, "energy": -2.000001}],
    })
    make_summary(traditional, "ga_seed0", {
        "random_seed": 0, "population_size": 20, "relaxations": 40,
        "best_energy_eV": -2.0,
        "steps": [{"step": 0, "best_energy_eV": -1.5},
                  {"step": 1, "best_energy_eV": -2.0}],
    })
    prefix = "agentic_without_unique_database_all14_four_surrogates_five_seeds_v1_"
    make_summary(surrogate, f"{prefix}bayesian_ridge_seed0", {
        "system": "Example", "surrogate_kind": "bayesian_ridge", "seed": 0,
        "best_energy_eV": -2.0, "stop_decision": {"done": True},
        "steps": [{"step": 0, "energy": -1, "occupancy": ["A", "B"]},
                  {"step": 1, "energy": -1.8, "occupancy": ["B", "A"]},
                  {"step": 2, "energy": -2, "occupancy": ["A", "A"]}],
    })
    rows = exporter.build_rows(traditional_root=traditional, surrogate_root=surrogate,
                               reference_root=reference.parent, systems=("Example",),
                               methods=("random", "ga", "bayesian_ridge"), seeds=(0,))
    by_method = {row["method"]: row for row in rows}
    assert len(rows) == 3
    assert by_method["random"]["first_reference_hit_step"] == 2
    assert by_method["ga"]["first_hit_generation_eval_min"] == 21
    assert by_method["ga"]["first_hit_generation_eval_max"] == 40
    assert by_method["ga"]["step_unit"] == "generation"
    assert by_method["bayesian_ridge"]["first_reference_hit_step"] == 2
    assert all(row["finite_reference_energy_eV"] == -2.0 for row in rows)
    output = tmp_path / "first_hits.csv"
    exporter.write_rows(output, rows)
    with output.open(newline="") as stream:
        readback = list(csv.DictReader(stream, delimiter=";"))
    assert len(readback) == 3
    assert readback[1]["first_hit_generation_eval_min"] == "21"


def test_missing_run_fails_without_creating_export(tmp_path):
    exporter = load_exporter()
    out = tmp_path / "not_written.csv"
    with pytest.raises(FileNotFoundError):
        rows = exporter.build_rows(traditional_root=tmp_path / "traditional",
                                   surrogate_root=tmp_path / "surrogates",
                                   reference_root=tmp_path / "references",
                                   systems=("Example",), methods=("random",), seeds=(0,))
        exporter.write_rows(out, rows)
    assert not out.exists()


def test_export_never_overwrites_existing_csv(tmp_path):
    exporter = load_exporter()
    out = tmp_path / "existing.csv"
    out.write_text("important prior result\n")
    with pytest.raises(FileExistsError):
        exporter.write_rows(out, [])
    assert out.read_text() == "important prior result\n"


def test_reported_best_must_match_trajectory_at_archived_energy_scale(tmp_path):
    exporter = load_exporter()
    reference = tmp_path / "refs" / "Example"
    reference.mkdir(parents=True)
    atoms = Atoms("H", positions=[(0, 0, 0)], cell=[5, 5, 5], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-1086.0, forces=np.zeros((1, 3)))
    write(reference / "optimized_all.extxyz", atoms)
    make_summary(tmp_path / "traditional", "random_seed0", {
        "best_energy_eV": -1086.000001,
        "steps": [{"step": 0, "energy": -1086.0}],
    })
    with pytest.raises(ValueError, match="best energy mismatch"):
        exporter.build_rows(traditional_root=tmp_path / "traditional",
                            surrogate_root=tmp_path / "surrogates",
                            reference_root=reference.parent,
                            systems=("Example",), methods=("random",), seeds=(0,))


def test_nonfinite_reference_energy_is_rejected(tmp_path):
    exporter = load_exporter()
    reference = tmp_path / "refs" / "Example"
    reference.mkdir(parents=True)
    atoms = Atoms("H", positions=[(0, 0, 0)], cell=[5, 5, 5], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=float("nan"), forces=np.zeros((1, 3)))
    write(reference / "optimized_all.extxyz", atoms)
    with pytest.raises(ValueError, match="nonfinite reference"):
        exporter.reference_energy(reference.parent, "Example")
