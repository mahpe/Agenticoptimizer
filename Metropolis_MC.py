"""
Metropolis Monte Carlo site-occupancy optimizer.

Fixed-composition swaps on lattice sites with MatCalc relaxation. Uphill moves
are accepted with a fixed probability (acceptance_rate); improving moves always
accept.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils import (
    DEFAULT_CONFIG,
    ROOT,
    apply_swap,
    evaluate_occupancy,
    fixed_rate_accept,
    initial_occupancy,
    load_config,
    occupancy_counts,
    print_loading_info,
    propose_swap_from_pairs,
    resolve_counts,
    resolve_optimization_input,
    setup_optimizer,
    split_lattice_and_framework,
    unlike_site_pairs,
    write_atoms_xyz,
    write_input_manifest,
    SwapStructureWriter,
)


@dataclass
class MCStepRecord:
    step: int
    energy: float
    occupancy: list[str]
    swap: list[int]
    delta_e: float
    accepted: bool
    acceptance_prob: float


@dataclass
class MCState:
    occupancy: list[str]
    energy: float
    history: list[MCStepRecord] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)  # configs already evaluated

    def occupancy_key(self, occupancy: list[str] | None = None) -> str:
        return "".join(occupancy if occupancy is not None else self.occupancy)


def propose_swap(occupancy: list[str], rng: np.random.Generator) -> tuple[int, int]:
    pairs = unlike_site_pairs(occupancy)
    if not pairs:
        raise RuntimeError("No unlike-site pairs available for swapping.")
    return propose_swap_from_pairs(pairs, rng)


def run_mc(cfg: dict, args: argparse.Namespace) -> MCState:
    relax = cfg.get("relax_params", {})
    mc_cfg = cfg.get("mc_params", {})

    opt_in = resolve_optimization_input(cfg)
    reference = opt_in.reference
    lattice_element = opt_in.lattice_element
    species = opt_in.atom_swaps
    ratio = opt_in.ratio
    vacancy_symbol = next((s for s in species if s == "X"), "X")

    num_steps = int(args.num_steps if args.num_steps is not None else mc_cfg.get("num_steps", 100))
    acceptance_rate = float(
        args.acceptance_rate
        if args.acceptance_rate is not None
        else mc_cfg.get("acceptance_rate", 0.3)
    )
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError(f"acceptance_rate must be in [0, 1], got {acceptance_rate}")
    seed = int(args.seed if args.seed is not None else mc_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    out_dir = Path(args.out_dir or mc_cfg.get("out_dir", "runs/mc_latest"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_input_manifest(out_dir, opt_in.manifest)
    xyz_writer = SwapStructureWriter(out_dir, species)

    print_loading_info(opt_in, cfg)
    template_sites, framework = split_lattice_and_framework(reference, lattice_element)
    n_sites = len(template_sites)
    counts = resolve_counts(n_sites, species, ratio)
    print(f"Lattice sites ({lattice_element}): {n_sites}")
    print(f"Target counts: {counts}")
    print(f"MC steps: {num_steps}, acceptance_rate: {acceptance_rate:.3f}, seed: {seed}")

    relax_kwargs = deepcopy(relax)
    relax_kwargs["traj_file"] = str(out_dir / "relaxation.xyz")
    optimizer = setup_optimizer(**relax_kwargs)

    occupancy = initial_occupancy(n_sites, species, ratio, rng)
    print(f"Initial occupancy: {''.join(occupancy)}")
    energy, _results, structure = evaluate_occupancy(
        occupancy, template_sites, framework, optimizer, vacancy_symbol
    )
    xyz_writer.write(structure, label="step_000_initial", occupancy=occupancy, energy=energy)
    print(f"Initial energy: {energy:.8f} eV")

    state = MCState(occupancy=occupancy, energy=energy)
    state.seen.add(state.occupancy_key())
    state.history.append(
        MCStepRecord(
            step=0,
            energy=energy,
            occupancy=occupancy.copy(),
            swap=[],
            delta_e=0.0,
            accepted=True,
            acceptance_prob=1.0,
        )
    )

    best_occ = occupancy.copy()
    best_energy = energy
    best_structure = structure
    write_atoms_xyz(out_dir / "best.xyz", best_structure)
    xyz_writer.write(best_structure, label="best_initial", occupancy=best_occ, energy=best_energy)

    n_accepted = 0
    n_proposed = 0

    # MC loop: propose unlike-site swap → relax → accept/reject (fixed uphill rate).
    for step in range(1, num_steps + 1):
        i, j = propose_swap(state.occupancy, rng)
        new_occ = apply_swap(state.occupancy, i, j)
        key = state.occupancy_key(new_occ)
        # Skip relax if we already visited this arrangement.
        if key in state.seen:
            state.history.append(
                MCStepRecord(
                    step=step,
                    energy=state.energy,
                    occupancy=state.occupancy.copy(),
                    swap=[i, j],
                    delta_e=0.0,
                    accepted=False,
                    acceptance_prob=0.0,
                )
            )
            continue

        n_proposed += 1
        new_energy, _results, new_structure = evaluate_occupancy(
            new_occ, template_sites, framework, optimizer, vacancy_symbol
        )
        delta_e = new_energy - state.energy
        accepted, accept_prob = fixed_rate_accept(delta_e, acceptance_rate, rng)

        record = MCStepRecord(
            step=step,
            energy=new_energy if accepted else state.energy,
            occupancy=new_occ.copy() if accepted else state.occupancy.copy(),
            swap=[i, j],
            delta_e=delta_e,
            accepted=accepted,
            acceptance_prob=accept_prob,
        )
        state.history.append(record)
        state.seen.add(key)  # mark proposed config even if rejected

        if accepted:
            n_accepted += 1
            state.occupancy = new_occ
            state.energy = new_energy
            xyz_writer.write(
                new_structure,
                label=f"step_{step:04d}",
                occupancy=new_occ,
                energy=new_energy,
            )
            print(
                f"Step {step:4d}: ACCEPT  E={new_energy:.8f} eV  dE={delta_e:+.6f}  swap=[{i},{j}]"
            )
            if new_energy < best_energy:
                best_energy = new_energy
                best_occ = new_occ.copy()
                best_structure = new_structure
                write_atoms_xyz(out_dir / "best.xyz", best_structure)
                xyz_writer.write(
                    best_structure,
                    label="best",
                    occupancy=best_occ,
                    energy=best_energy,
                )
                print("           new best")
        else:
            # Keep current state; proposal only logged in history.
            print(
                f"Step {step:4d}: reject  E={state.energy:.8f} eV  dE={delta_e:+.6f}  "
                f"P_acc={accept_prob:.4f}  swap=[{i},{j}]"
            )

    summary = {
        "best_energy_eV": best_energy,
        "best_occupancy": best_occ,
        "best_counts": occupancy_counts(best_occ),
        "acceptance_rate_target": acceptance_rate,
        "num_steps": num_steps,
        "n_proposed": n_proposed,
        "n_accepted": n_accepted,
        "acceptance_rate": n_accepted / n_proposed if n_proposed else 0.0,
        "final_energy_eV": state.energy,
        "final_occupancy": state.occupancy,
        "steps": [
            {
                "step": r.step,
                "energy": r.energy,
                "occupancy": r.occupancy,
                "swap": r.swap,
                "delta_e": r.delta_e,
                "accepted": r.accepted,
                "acceptance_prob": r.acceptance_prob,
            }
            for r in state.history
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_atoms_xyz(out_dir / "best.xyz", best_structure)
    xyz_writer.write(best_structure, label="best_final", occupancy=best_occ, energy=best_energy)

    print(f"\nDone. Best energy {best_energy:.8f} eV")
    print(f"Accepted {n_accepted}/{n_proposed} proposed moves "
          f"({summary['acceptance_rate']:.1%})")
    print(f"Structure trajectory: {xyz_writer.trajectory}")
    print(f"Results written to {out_dir}")
    return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monte Carlo site optimization with fixed uphill acceptance rate."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.toml")
    p.add_argument("--num-steps", type=int, default=None, help="Number of MC steps")
    p.add_argument("--acceptance-rate", type=float, default=None, help="Fixed uphill acceptance rate in [0, 1]")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_mc(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
