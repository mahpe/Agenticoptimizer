"""
Metropolis Monte Carlo site-occupancy optimizer.

Search space: which species sit on the lattice sites of a fixed framework
(e.g. Na vs vacancy on an olivine host). Composition is held fixed: every
proposal exchanges two sites that currently hold different species, so the
counts never change.

Each new arrangement is relaxed with MatCalc and scored by total energy.
Downhill moves (delta_E <= 0) are always accepted. Uphill moves are accepted
with a constant probability ``acceptance_rate``, not a temperature-dependent
Boltzmann factor. The best energy seen so far is tracked separately from the
current walker, so a rejected or uphill move does not overwrite the optimum.
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
    """One row of the trajectory written to summary.json."""

    step: int
    energy: float  # walker energy after this step (unchanged on reject)
    occupancy: list[str]  # walker occupancy after this step
    swap: list[int]  # site indices exchanged; empty for the initial step
    delta_e: float  # proposed_energy - current_energy; 0 if the proposal was skipped
    accepted: bool
    acceptance_prob: float  # 1 for downhill, acceptance_rate for uphill, 0 if skipped


@dataclass
class MCState:
    """Current walker plus the configurations already sent to relaxation."""

    occupancy: list[str]
    energy: float
    history: list[MCStepRecord] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)  # configs already evaluated

    def occupancy_key(self, occupancy: list[str] | None = None) -> str:
        """Canonical string key. Order is site order, so permutations compare equal only if identical."""
        return "".join(occupancy if occupancy is not None else self.occupancy)


def propose_swap(occupancy: list[str], rng: np.random.Generator) -> tuple[int, int]:
    """Pick two sites that hold different species, uniformly among such pairs.

    Swapping like sites would leave the arrangement unchanged and waste a
    relaxation, so those pairs are excluded.
    """
    pairs = unlike_site_pairs(occupancy)
    if not pairs:
        raise RuntimeError("No unlike-site pairs available for swapping.")
    return propose_swap_from_pairs(pairs, rng)


def run_mc(cfg: dict, args: argparse.Namespace) -> MCState:
    relax = cfg.get("relax_params", {})
    mc_cfg = cfg.get("mc_params", {})

    # Host structure plus which symbols may occupy the lattice sites.
    # CLI flags override the matching keys in [mc_params].
    opt_in = resolve_optimization_input(cfg)
    reference = opt_in.reference
    lattice_element = opt_in.lattice_element
    species = opt_in.atom_swaps
    ratio = opt_in.ratio
    # "X" is a vacancy placeholder. It is stripped before relaxation so the
    # calculator only sees real atoms.
    vacancy_symbol = next((s for s in species if s == "X"), "X")

    num_steps = int(args.num_steps if args.num_steps is not None else mc_cfg.get("num_steps", 100))
    # Probability of accepting a move that raises the energy. Downhill moves
    # are accepted regardless of this value.
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
    # Lattice sites are the only positions that get swapped. The framework
    # (everything else, e.g. FePO4) is held fixed and reattached at each step.
    template_sites, framework = split_lattice_and_framework(reference, lattice_element)
    n_sites = len(template_sites)
    counts = resolve_counts(n_sites, species, ratio)
    print(f"Lattice sites ({lattice_element}): {n_sites}")
    print(f"Target counts: {counts}")
    print(f"MC steps: {num_steps}, acceptance_rate: {acceptance_rate:.3f}, seed: {seed}")

    relax_kwargs = deepcopy(relax)
    relax_kwargs["traj_file"] = str(out_dir / "relaxation.xyz")
    optimizer = setup_optimizer(**relax_kwargs)

    # Random starting arrangement at the target composition, then one relaxation.
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

    # Best-so-far is independent of the chain: an accepted uphill move
    # updates ``state`` but leaves ``best_*`` unchanged.
    best_occ = occupancy.copy()
    best_energy = energy
    best_structure = structure
    write_atoms_xyz(out_dir / "best.xyz", best_structure)
    xyz_writer.write(best_structure, label="best_initial", occupancy=best_occ, energy=best_energy)

    n_accepted = 0
    n_proposed = 0  # unique arrangements that were actually relaxed

    # Propose an unlike-site swap, relax if unseen, then accept or reject.
    for step in range(1, num_steps + 1):
        i, j = propose_swap(state.occupancy, rng)
        new_occ = apply_swap(state.occupancy, i, j)
        key = state.occupancy_key(new_occ)
        # Already scored this arrangement. Log the attempt and stay put
        # without calling the relaxer again.
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
        # delta_e <= 0 always accepts. Uphill moves accept with acceptance_rate.
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
        # Record the proposal even when it is rejected, so a later step
        # that lands on the same arrangement skips the relaxation.
        state.seen.add(key)

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
            # Only a new global minimum overwrites best.xyz.
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
            # Chain stays on the previous occupancy; the proposal is only in history.
            print(
                f"Step {step:4d}: reject  E={state.energy:.8f} eV  dE={delta_e:+.6f}  "
                f"P_acc={accept_prob:.4f}  swap=[{i},{j}]"
            )

    # acceptance_rate_target is the uphill probability requested by the user.
    # acceptance_rate is the fraction of unique proposals that were accepted,
    # which is higher because downhill moves always count as accepted.
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
