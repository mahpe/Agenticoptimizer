"""
Agentic site-occupancy optimizer.

Mirrors the cPaiNN GA workflow (lattice sites on a framework, relax, score)
but replaces GA operators with an agent that proposes site swaps using Skill.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.io import write

from Main_scripts import (
    apply_occupancy,
    assemble_structure,
    get_reference_structures,
    get_score,
    setup_optimizer,
    split_lattice_and_framework,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_SKILL = ROOT / "Skill.md"


@dataclass
class StepRecord:
    step: int
    energy: float
    occupancy: list[str]
    swaps: list[list[int]]
    rationale: str = ""


@dataclass
class OptimizerState:
    occupancy: list[str]
    energy: float
    history: list[StepRecord] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)

    def occupancy_key(self, occupancy: list[str] | None = None) -> str:
        return "".join(occupancy if occupancy is not None else self.occupancy)


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    # Normalize top-level + tables into a flat-friendly dict.
    if "supercell_size" in cfg and isinstance(cfg["supercell_size"], list):
        cfg["supercell_size"] = tuple(cfg["supercell_size"])
    return cfg


def resolve_counts(n_sites: int, species: list[str], ratio: list[float]) -> dict[str, int]:
    if len(species) != len(ratio):
        raise ValueError("atom_swaps and ratio must have the same length")
    if abs(sum(ratio) - 1.0) > 1e-6:
        raise ValueError(f"ratio must sum to 1, got {sum(ratio)}")

    raw = [r * n_sites for r in ratio]
    counts = [int(np.floor(x)) for x in raw]
    # Distribute remaining seats by largest remainder.
    remainders = sorted(
        ((raw[i] - counts[i], i) for i in range(len(species))),
        reverse=True,
    )
    leftover = n_sites - sum(counts)
    for k in range(leftover):
        counts[remainders[k][1]] += 1
    return {species[i]: counts[i] for i in range(len(species))}


def initial_occupancy(
    n_sites: int,
    species: list[str],
    ratio: list[float],
    rng: np.random.Generator,
) -> list[str]:
    counts = resolve_counts(n_sites, species, ratio)
    occ = []
    for sym, n in counts.items():
        occ.extend([sym] * n)
    rng.shuffle(occ)
    return occ


def apply_swaps(occupancy: list[str], swaps: list[list[int]]) -> list[str]:
    occ = occupancy.copy()
    n = len(occ)
    for pair in swaps:
        if len(pair) != 2:
            raise ValueError(f"Swap must be [i, j], got {pair}")
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Swap indices out of range: {pair} for N={n}")
        occ[i], occ[j] = occ[j], occ[i]
    return occ


def occupancy_counts(occupancy: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym in occupancy:
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def extract_json(text: str) -> dict:
    """Parse the first JSON object from an agent reply."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from agent reply:\n{text[:500]}")


class SkillAgent:
    """Agent that proposes swaps according to Skill.md."""

    def __init__(self, skill_text: str, backend: str = "cursor", model: str = "composer-2.5"):
        self.skill_text = skill_text
        self.backend = backend
        self.model = model
        self._cursor_agent = None
        self._rng = np.random.default_rng(0)

    def close(self):
        if self._cursor_agent is not None:
            try:
                self._cursor_agent.close()
            except Exception:
                pass
            self._cursor_agent = None

    def _ensure_cursor(self):
        if self._cursor_agent is not None:
            return
        try:
            from cursor_sdk import Agent, LocalAgentOptions
        except ImportError as exc:
            raise ImportError(
                "cursor-sdk is required for --agent cursor. "
                "Install with: pip install cursor-sdk"
            ) from exc

        api_key = os.environ.get("CURSOR_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set CURSOR_API_KEY to use the Cursor agent backend."
            )

        self._cursor_agent = Agent.create(
            model=self.model,
            api_key=api_key,
            local=LocalAgentOptions(cwd=str(ROOT)),
        )

    def propose(self, state: OptimizerState, species: list[str], max_swaps: int) -> dict:
        if self.backend == "random":
            return self._propose_random(state, max_swaps)
        if self.backend == "cursor":
            return self._propose_cursor(state, species, max_swaps)
        raise ValueError(f"Unknown agent backend: {self.backend}")

    def _propose_random(self, state: OptimizerState, max_swaps: int) -> dict:
        occ = state.occupancy
        # Pick pairs of unlike sites.
        unlike = [
            (i, j)
            for i in range(len(occ))
            for j in range(i + 1, len(occ))
            if occ[i] != occ[j]
        ]
        if not unlike:
            return {"swaps": [], "done": True, "rationale": "no unlike pairs"}
        n = int(self._rng.integers(1, min(max_swaps, len(unlike)) + 1))
        picks = self._rng.choice(len(unlike), size=n, replace=False)
        swaps = [list(unlike[int(k)]) for k in np.atleast_1d(picks)]
        return {"swaps": swaps, "rationale": "random unlike-site swaps", "done": False}

    def _propose_cursor(self, state: OptimizerState, species: list[str], max_swaps: int) -> dict:
        self._ensure_cursor()
        prompt = self._build_prompt(state, species, max_swaps)
        run = self._cursor_agent.send(prompt)
        result = run.wait()
        if getattr(result, "status", None) == "error":
            raise RuntimeError(f"Cursor agent run failed: {getattr(result, 'id', result)}")

        text = getattr(result, "result", None)
        if not text:
            # Fall back to collecting assistant text from messages.
            chunks = []
            for message in run.messages():
                if getattr(message, "type", None) == "assistant":
                    content = getattr(getattr(message, "message", None), "content", []) or []
                    for block in content:
                        if getattr(block, "type", None) == "text":
                            chunks.append(block.text)
            text = "".join(chunks)
        return extract_json(text)

    def _build_prompt(self, state: OptimizerState, species: list[str], max_swaps: int) -> str:
        recent = state.history[-8:]
        history_lines = [
            f"  step {r.step}: E={r.energy:.6f} eV, swaps={r.swaps}, occ={''.join(r.occupancy)}"
            for r in recent
        ]
        history_block = "\n".join(history_lines) if history_lines else "  (none yet)"
        return f"""{self.skill_text}

## Current optimization state

- Number of lattice sites: {len(state.occupancy)}
- Species: {species}
- Counts: {occupancy_counts(state.occupancy)}
- Current energy: {state.energy:.8f} eV
- Current occupancy (index → symbol): {state.occupancy}
- Max swaps this step: {max_swaps}

## Recent history
{history_block}

Propose the next swap move now. Reply with JSON only.
"""


def evaluate_occupancy(
    occupancy: list[str],
    template_sites,
    framework,
    optimizer,
    vacancy_symbol: str = "X",
):
    sites = apply_occupancy(template_sites, occupancy)
    structure = assemble_structure(sites, framework, vacancy_symbol=vacancy_symbol)
    energy, results = get_score(structure, optimizer)
    return energy, results, structure


def run_optimization(cfg: dict, args: argparse.Namespace) -> OptimizerState:
    swap = cfg.get("swap_params", {})
    relax = cfg.get("relax_params", {})
    agent_cfg = cfg.get("agent_params", {})

    lattice_element = str(swap.get("lattice_sites", "Na"))
    species = [str(s) for s in swap.get("atom_swaps", ["Na", "X"])]
    ratio = [float(x) for x in swap.get("ratio", [0.66, 0.34])]
    vacancy_symbol = next((s for s in species if s == "X"), "X")

    cif_path = Path(cfg.get("cif_path", ROOT / "NaFePO4_olivine_new.cif"))
    supercell_size = tuple(cfg.get("supercell_size", (1, 1, 1)))
    out_dir = Path(args.out_dir or agent_cfg.get("out_dir", "runs/latest"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_iters = int(args.max_iters or agent_cfg.get("max_iters", 20))
    max_swaps = int(agent_cfg.get("max_swaps_per_step", 4))
    seed = int(args.seed if args.seed is not None else agent_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    print(f"Loading structure from {cif_path} with supercell {supercell_size}")
    reference = get_reference_structures(str(cif_path), supercell_size=supercell_size)
    template_sites, framework = split_lattice_and_framework(reference, lattice_element)
    n_sites = len(template_sites)
    counts = resolve_counts(n_sites, species, ratio)
    print(f"Lattice sites ({lattice_element}): {n_sites}")
    print(f"Target counts: {counts}")

    relax_kwargs = deepcopy(relax)
    # Unique traj per run directory to avoid collisions.
    relax_kwargs["traj_file"] = str(out_dir / "relaxation.traj")
    optimizer = setup_optimizer(**relax_kwargs)

    occupancy = initial_occupancy(n_sites, species, ratio, rng)
    print(f"Initial occupancy: {''.join(occupancy)}")
    energy, results, structure = evaluate_occupancy(
        occupancy, template_sites, framework, optimizer, vacancy_symbol
    )
    write(out_dir / "step_000_initial.cif", structure)
    print(f"Initial energy: {energy:.8f} eV")

    state = OptimizerState(occupancy=occupancy, energy=energy)
    state.seen.add(state.occupancy_key())
    state.history.append(
        StepRecord(step=0, energy=energy, occupancy=occupancy.copy(), swaps=[], rationale="initial")
    )

    best_occ = occupancy.copy()
    best_energy = energy
    best_structure = structure

    skill_path = Path(args.skill or agent_cfg.get("skill_path", DEFAULT_SKILL))
    if not skill_path.is_absolute():
        skill_path = ROOT / skill_path
    skill_text = skill_path.read_text(encoding="utf-8")
    agent = SkillAgent(
        skill_text=skill_text,
        backend=args.agent,
        model=args.model or agent_cfg.get("model", "composer-2.5"),
    )

    try:
        for step in range(1, max_iters + 1):
            print(f"\n=== Step {step}/{max_iters} ===")
            proposal = agent.propose(state, species, max_swaps)
            swaps = proposal.get("swaps") or []
            rationale = str(proposal.get("rationale", ""))
            print(f"Agent proposal: swaps={swaps} | {rationale}")

            if proposal.get("done") and not swaps:
                print("Agent signalled done.")
                break

            if not swaps:
                print("Empty swap list; stopping.")
                break

            new_occ = apply_swaps(state.occupancy, swaps)
            if occupancy_counts(new_occ) != occupancy_counts(state.occupancy):
                print("Rejected: proposal changed composition.")
                continue

            key = state.occupancy_key(new_occ)
            if key in state.seen:
                print("Rejected: duplicate occupancy.")
                continue

            energy, results, structure = evaluate_occupancy(
                new_occ, template_sites, framework, optimizer, vacancy_symbol
            )
            state.seen.add(key)
            state.occupancy = new_occ
            state.energy = energy
            state.history.append(
                StepRecord(
                    step=step,
                    energy=energy,
                    occupancy=new_occ.copy(),
                    swaps=swaps,
                    rationale=rationale,
                )
            )
            write(out_dir / f"step_{step:03d}.cif", structure)
            print(f"Energy: {energy:.8f} eV  (best {best_energy:.8f} eV)")

            if energy < best_energy:
                best_energy = energy
                best_occ = new_occ.copy()
                best_structure = structure
                write(out_dir / "best.cif", best_structure)
                print("New best configuration.")
    finally:
        agent.close()

    # Persist summary
    summary = {
        "best_energy_eV": best_energy,
        "best_occupancy": best_occ,
        "best_counts": occupancy_counts(best_occ),
        "steps": [
            {
                "step": r.step,
                "energy": r.energy,
                "occupancy": r.occupancy,
                "swaps": r.swaps,
                "rationale": r.rationale,
            }
            for r in state.history
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write(out_dir / "best.cif", best_structure)
    print(f"\nDone. Best energy {best_energy:.8f} eV")
    print(f"Results written to {out_dir}")
    return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic atomistic site optimization (GA-style scoring, agent swaps)."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.toml",
    )
    p.add_argument(
        "--skill",
        type=Path,
        default=None,
        help="Path to Skill.md (default: ./Skill.md or agent_params.skill_path)",
    )
    p.add_argument(
        "--agent",
        choices=("cursor", "random"),
        default="cursor",
        help="Swap proposer: Cursor SDK agent (Skill.md) or random baseline",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Cursor model id (cursor backend only)",
    )
    p.add_argument("--max-iters", type=int, default=None, help="Max agent steps")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for initial occupancy")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for trajectories, CIFs, and summary.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_optimization(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
