"""
Genetic-algorithm site-occupancy optimizer.

Fixed-composition lattice-site search using ASE-GA slab operators and
MatCalc total-energy relaxation/scoring via utils.py (same as MC/agent).
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import Atoms

try:
    from ase_ga import set_raw_score
    from ase_ga.data import DataConnection, PrepareDB
    from ase_ga.offspring_creator import OperationSelector
    from ase_ga.population import RankFitnessPopulation
    from ase_ga.slab_operators import (
        CutSpliceSlabCrossover,
        RandomCompositionMutation,
        RandomSlabPermutation,
        SymmetrySlabPermutation,
    )
except ImportError:  # pragma: no cover - older ASE bundles ase.ga
    from ase.ga import set_raw_score
    from ase.ga.data import DataConnection, PrepareDB
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import RankFitnessPopulation
    from ase.ga.slab_operators import (
        CutSpliceSlabCrossover,
        RandomCompositionMutation,
        RandomSlabPermutation,
        SymmetrySlabPermutation,
    )

from utils import (
    DEFAULT_CONFIG,
    ROOT,
    initial_occupancy,
    lattice_only,
    load_config,
    print_loading_info,
    relax_lattice_sites,
    resolve_optimization_input,
    setup_optimizer,
    split_lattice_and_framework,
    vacancy_symbol_from_swaps,
    write_atoms_xyz,
    write_input_manifest,
)


def run_ga(cfg: dict, args: argparse.Namespace) -> None:
    relax = cfg.get("relax_params", {})
    ga_cfg = cfg.get("ga_params", {})

    opt_in = resolve_optimization_input(cfg)
    reference = opt_in.reference
    lattice_element = opt_in.lattice_element
    metals = opt_in.atom_swaps
    ratio = opt_in.ratio
    vacancy_symbol = vacancy_symbol_from_swaps(metals)
    assemble_vacancy = vacancy_symbol or "X"

    num_gens = int(args.num_gens if args.num_gens is not None else ga_cfg.get("num_gens", 400))
    pop_size = int(args.pop_size if args.pop_size is not None else ga_cfg.get("pop_size", 200))
    random_seed = int(
        args.seed if args.seed is not None else ga_cfg.get("random_seed", ga_cfg.get("seed", 0))
    )

    out_dir = Path(args.out_dir or ga_cfg.get("out_dir", "runs/ga_latest"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(random_seed)

    print_loading_info(opt_in, cfg)
    write_input_manifest(out_dir, opt_in.manifest)

    _template_sites, framework = split_lattice_and_framework(reference, lattice_element)
    lattice_template = lattice_only(reference, lattice_element)
    fu = len(lattice_template)

    relax_kwargs = deepcopy(relax)
    relax_kwargs["traj_file"] = str(out_dir / "relaxation.traj")
    optimizer = setup_optimizer(**relax_kwargs)

    lattice_constants = reference.cell.cellpar()
    stem = Path(str(opt_in.manifest.get("input_cif", "structure"))).stem
    db_name = out_dir / (
        f"{stem}_{opt_in.min_cell_size if opt_in.prepared else 'manual'}"
        f"_{random_seed}_{pop_size}_{relax.get('fmax', 0.2)}_GA.db"
    )

    print(f"Supercell size: {opt_in.supercell_size if not opt_in.prepared else (opt_in.min_cell_size,) * 3}")
    print(f"Swap sites ({lattice_element}): {fu}")
    print(f"Species: {metals}, ratio: {ratio}")
    print(f"Lattice constants: {lattice_constants}")
    print(f"Population size: {pop_size}")
    print(f"Generations: {num_gens}")
    print(f"Random seed: {random_seed}")
    print(f"Database: {db_name}")

    if db_name.exists():
        db_name.unlink()

    # ASE-GA database: stores candidates, energies, and population metadata.
    db = PrepareDB(
        db_name,
        population_size=pop_size,
        metals=metals,
        lattice_constants=lattice_constants,
        formula_unit=fu,
    )

    # Seed population with random permutations at fixed composition.
    for i in range(pop_size):
        new_structure = lattice_template.copy()
        occ = initial_occupancy(fu, metals, ratio, np.random.default_rng(i))
        for index, symbol in enumerate(occ):
            new_structure[index].symbol = symbol
        atoms_string = "".join(new_structure.get_chemical_symbols())
        db.add_unrelaxed_candidate(new_structure, atoms_string=atoms_string)

    db = DataConnection(str(db_name))
    pop_size = db.get_param("population_size")
    fu = db.get_param("formula_unit")
    metals = db.get_param("metals")

    relax_counter = {"n": 0}
    best = {"energy": float("inf"), "atoms": None}
    gen_history: list[dict] = []

    def get_comp(atoms: Atoms) -> str:
        return atoms.get_chemical_formula()

    def get_total_energy(atoms: Atoms) -> float:
        relax_counter["n"] += 1
        e_total, relaxed = relax_lattice_sites(
            atoms, framework, optimizer, vacancy_symbol=assemble_vacancy
        )
        atoms.info["energy"] = e_total
        if e_total < best["energy"]:
            best["energy"] = e_total
            best["atoms"] = relaxed
        return e_total

    oclist = [
        (3, CutSpliceSlabCrossover(rng=rng)),
        (1, RandomSlabPermutation(rng=rng)),
        (1, RandomCompositionMutation(rng=rng)),
        (1, SymmetrySlabPermutation(rng=rng)),
    ]  # weighted slab GA operators from ase-ga
    operation_selector = OperationSelector(*zip(*oclist), rng=rng)
    pop = RankFitnessPopulation(
        data_connection=db,
        population_size=pop_size,
        variable_function=get_comp,
    )

    print("Evaluating initial candidates...")
    while db.get_number_of_unrelaxed_candidates() > 0:
        candidate = db.get_an_unrelaxed_candidate()
        e_total = get_total_energy(candidate)
        set_raw_score(candidate, -e_total)
        candidate.info.setdefault("key_value_pairs", {})
        candidate.info["key_value_pairs"]["E_tot"] = e_total
        db.add_relaxed_step(candidate)
    pop.update()
    gen_history.append({"step": 0, "best_energy_eV": best["energy"]})
    (out_dir / "summary_partial.json").write_text(
        json.dumps({"steps": gen_history}, indent=2), encoding="utf-8"
    )

    gen_num = db.get_generation_number()
    for gen_i in range(num_gens):
        print(f"Creating and evaluating generation {gen_num + gen_i}")
        new_generation = []
        for _ in range(pop_size):
            dup = True
            # Resample until ASE-GA produces a novel composition string.
            while dup:
                parents = pop.get_two_candidates()
                op = operation_selector.get_operator()
                offspring, _desc = op.get_new_individual(parents)
                if offspring is None:
                    continue
                atoms_string = "".join(offspring.get_chemical_symbols())
                dup = db.is_duplicate(atoms_string=atoms_string)
            try:
                e_total = get_total_energy(offspring)
                set_raw_score(offspring, -e_total)
            except Exception as exc:
                print(f"Failing: {exc}")
                continue
            offspring.info.setdefault("key_value_pairs", {})
            offspring.info["key_value_pairs"]["E_tot"] = e_total
            new_generation.append(offspring)

        db.add_more_relaxed_candidates(new_generation)
        pop.update()
        gen_history.append({"step": gen_i + 1, "best_energy_eV": best["energy"]})
        (out_dir / "summary_partial.json").write_text(
            json.dumps({"steps": gen_history}, indent=2), encoding="utf-8"
        )

    summary = {
        "best_energy_eV": best["energy"] if best["energy"] != float("inf") else None,
        "formula_unit": fu,
        "population_size": pop_size,
        "generations": num_gens,
        "random_seed": random_seed,
        "database": str(db_name),
        "relaxations": relax_counter["n"],
        "steps": gen_history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best["atoms"] is not None:
        write_atoms_xyz(out_dir / "best.xyz", best["atoms"])
    print(f"\nDone. {relax_counter['n']} relaxations performed.")
    if best["energy"] != float("inf"):
        print(f"Best energy {best['energy']:.8f} eV")
    print(f"Results written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genetic-algorithm site optimization (cPaiNN GA + MatCalc relax)."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.toml")
    p.add_argument("--num-gens", type=int, default=None, help="Number of GA generations")
    p.add_argument("--pop-size", type=int, default=None, help="Population size")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_ga(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
