"""
Agentic site-occupancy optimizer.

Fixed-composition swaps on configurable lattice sites (any species from
config.toml / prepared disordered CIF). An agent guided by Skill.md proposes
site exchanges; structures are relaxed and scored with MatCalc via utils.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.io import write

from utils import (
    DEFAULT_CONFIG,
    ROOT,
    apply_swaps,
    evaluate_occupancy,
    initial_occupancy,
    load_config,
    occupancy_counts,
    print_loading_info,
    resolve_counts,
    resolve_optimization_input,
    setup_optimizer,
    split_lattice_and_framework,
    vacancy_symbol_from_swaps,
    write_atoms_xyz,
    write_input_manifest,
    SwapStructureWriter,
)

DEFAULT_SKILL = ROOT / "Skill.md"
# openai: Hermes/Ollama/vLLM/OpenRouter via chat-completions; cursor: optional SDK
AGENT_BACKENDS = ("random", "openai", "cursor")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


@dataclass
class StepRecord:
    """One accepted or evaluated proposal: energy, occupancy, and the swaps that produced it."""

    step: int
    energy: float
    occupancy: list[str]
    swaps: list[list[int]]  # each inner list is a pair of site indices
    rationale: str = ""  # short agent explanation; empty for the random backend


@dataclass
class OptimizerState:
    """Walker used by the live agent and the random-swap baseline.

    ``seen`` stores occupancy strings that have already been relaxed, so a
    repeated arrangement is rejected instead of spending another MatCalc call.
    ``last_rejects`` is fed back into the next prompt so the agent does not
    immediately repeat a move the runtime refused.
    """

    occupancy: list[str]
    energy: float
    history: list[StepRecord] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    last_rejects: list[str] = field(default_factory=list)

    def occupancy_key(self, occupancy: list[str] | None = None) -> str:
        """Join site symbols in lattice order. Two keys match only when every site matches."""
        return "".join(occupancy if occupancy is not None else self.occupancy)


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
    """Propose fixed-composition swap moves using Skill.md + chosen LLM backend."""

    def __init__(
        self,
        skill_text: str,
        backend: str = "openai",
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        cursor_api_key: str | None = None,
        api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV,
        temperature: float = 0.2,
    ):
        self.skill_text = skill_text
        self.backend = backend
        self.model = model or ("composer-2.5" if backend == "cursor" else "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        self.cursor_api_key = cursor_api_key or os.environ.get("CURSOR_API_KEY")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self._cursor_agent = None
        self._rng = np.random.default_rng(0)
        self._chat_history: list[dict[str, str]] = []  # multi-turn context for openai backend

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

        if not self.cursor_api_key:
            raise EnvironmentError(
                "Set cursor_api_key in config.toml [agent_params], CURSOR_API_KEY, or pass --cursor-api-key."
            )

        self._cursor_agent = Agent.create(
            model=self.model,
            api_key=self.cursor_api_key,
            local=LocalAgentOptions(cwd=str(ROOT)),
        )

    def propose(
        self,
        state: OptimizerState,
        species: list[str],
        ratio: list[float],
        target_counts: dict[str, int],
        lattice_element: str,
        vacancy_symbol: str | None,
        max_swaps: int,
    ) -> dict:
        if self.backend == "random":
            return self._propose_random(state, max_swaps)
        if self.backend == "openai":
            return self._propose_openai(
                state, species, ratio, target_counts, lattice_element, vacancy_symbol, max_swaps
            )
        if self.backend == "cursor":
            return self._propose_cursor(
                state, species, ratio, target_counts, lattice_element, vacancy_symbol, max_swaps
            )
        raise ValueError(
            f"Unknown agent backend: {self.backend!r}. "
            f"Choose from: {', '.join(AGENT_BACKENDS)}"
        )

    def _propose_openai(
        self,
        state: OptimizerState,
        species: list[str],
        ratio: list[float],
        target_counts: dict[str, int],
        lattice_element: str,
        vacancy_symbol: str | None,
        max_swaps: int,
    ) -> dict:
        if not self.api_key:
            raise EnvironmentError(
                f"Set {self.api_key_env} or OPENAI_API_KEY for the openai agent backend."
            )

        user_prompt = self._build_prompt(
            state,
            species,
            ratio,
            target_counts,
            lattice_element,
            vacancy_symbol,
            max_swaps,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.skill_text}\n\n"
                    "You are the swap proposer for an atomistic optimizer. "
                    "Follow Skill.md exactly and reply with a single JSON object only."
                ),
            },
            *self._chat_history,
            {"role": "user", "content": user_prompt},
        ]
        text = self._openai_chat_completion(messages)
        proposal = extract_json(text)
        # Keep prior turns so the model can learn from accepted/rejected moves.
        self._chat_history.append({"role": "user", "content": user_prompt})
        self._chat_history.append({"role": "assistant", "content": text})
        return proposal

    def _openai_chat_completion(self, messages: list[dict[str, str]]) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for --agent openai. Install with: pip install httpx"
            ) from exc

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        data = None
        last_response = None
        for attempt in range(8):
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, headers=headers, json=payload)
            last_response = response
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else float(min(2 ** attempt, 60))
                except ValueError:
                    wait = float(min(2 ** attempt, 60))
                print(
                    f"OpenAI HTTP {response.status_code}; retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/8)"
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            break
        if data is None:
            last_response.raise_for_status()
            raise RuntimeError("OpenAI-compatible API request failed after retries")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI-compatible API returned no choices: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            chunks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(block.get("text", ""))
            content = "".join(chunks)
        if not content or not str(content).strip():
            raise RuntimeError(f"OpenAI-compatible API returned empty content: {data}")
        return str(content).strip()

    def _propose_random(self, state: OptimizerState, max_swaps: int) -> dict:
        occ = state.occupancy
        # Only unlike-site pairs change the arrangement.
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

    def _propose_cursor(
        self,
        state: OptimizerState,
        species: list[str],
        ratio: list[float],
        target_counts: dict[str, int],
        lattice_element: str,
        vacancy_symbol: str | None,
        max_swaps: int,
    ) -> dict:
        self._ensure_cursor()
        prompt = self._build_prompt(
            state,
            species,
            ratio,
            target_counts,
            lattice_element,
            vacancy_symbol,
            max_swaps,
        )
        run = self._cursor_agent.send(prompt)
        result = run.wait()
        if getattr(result, "status", None) == "error":
            raise RuntimeError(f"Cursor agent run failed: {getattr(result, 'id', result)}")

        text = getattr(result, "result", None)
        if not text:
            chunks = []
            for message in run.messages():
                if getattr(message, "type", None) == "assistant":
                    content = getattr(getattr(message, "message", None), "content", []) or []
                    for block in content:
                        if getattr(block, "type", None) == "text":
                            chunks.append(block.text)
            text = "".join(chunks)
        return extract_json(text)

    def _build_prompt(
        self,
        state: OptimizerState,
        species: list[str],
        ratio: list[float],
        target_counts: dict[str, int],
        lattice_element: str,
        vacancy_symbol: str | None,
        max_swaps: int,
    ) -> str:
        recent = state.history[-8:]
        history_lines = [
            f"  step {r.step}: E={r.energy:.6f} eV, swaps={r.swaps}, occ={''.join(r.occupancy)}"
            for r in recent
        ]
        history_block = "\n".join(history_lines) if history_lines else "  (none yet)"
        reject_lines = [f"  {line}" for line in state.last_rejects[-6:]]
        reject_block = "\n".join(reject_lines) if reject_lines else "  (none)"
        best = min(state.history, key=lambda r: r.energy) if state.history else None
        if best is None:
            best_block = "  (none yet)"
            on_best = "yes (initial)"
        else:
            on_best = (
                "yes — search unlike-site neighbors of this occupancy"
                if state.occupancy == best.occupancy
                else "NO — the only legal move is to restore the best occupancy"
            )
            best_block = (
                f"  step {best.step}: E={best.energy:.8f} eV, "
                f"occ={''.join(best.occupancy)}"
            )

        ratio_parts = [
            f"{sym}={target_counts.get(sym, 0)} ({ratio[i]:.4f})"
            for i, sym in enumerate(species)
        ]
        vacancy_note = (
            f"`{vacancy_symbol}` marks a vacancy (removed before relaxation)."
            if vacancy_symbol
            else "No vacancy species in this system — all swap sites stay fully occupied."
        )
        return f"""{self.skill_text}

## System configuration (from config.toml)

- Swap-site placeholder element: {lattice_element}
- Allowed species (`atom_swaps`): {species}
- Target composition: {", ".join(ratio_parts)}
- {vacancy_note}
- Number of swap sites: {len(state.occupancy)}

## Current optimization state

- Current counts: {occupancy_counts(state.occupancy)} (must match target)
- Current energy: {state.energy:.8f} eV
- Current occupancy (index → symbol): {state.occupancy}
- Currently on the best occupancy: {on_best}
- Best energy so far (runtime keeps this structure):
{best_block}
- Max swaps this step: {max_swaps}

## Recent history
{history_block}

## Recent rejected proposals (not evaluated; do not repeat)
{reject_block}

Propose the next swap. Follow Skill.md stop rules strictly: do **not** set
`"done": true` until you have searched neighbors of the **best** occupancy
and met the minimum unsuccessful-attempt / motif-diversity requirements.
Reply with JSON only.
"""


def run_agent(cfg: dict, args: argparse.Namespace) -> OptimizerState:
    """Random-swap baseline or live Skill.md agent, selected by ``--agent``.

    Both backends share the same relaxation, fixed composition, and duplicate
    rejection. ``random`` draws unlike-site swaps locally. ``openai`` and
    ``cursor`` ask a model for the next swaps and parse a single JSON object.
    The search stops at ``max_iters`` proposals after the initial structure,
    or earlier if the agent returns ``done: true``.
    """
    relax = cfg.get("relax_params", {})
    agent_cfg = cfg.get("agent_params", {})

    # Load structure + swap metadata (auto from disordered CIF or prepared manifest).
    opt_in = resolve_optimization_input(cfg)
    reference = opt_in.reference
    lattice_element = opt_in.lattice_element
    species = opt_in.atom_swaps
    ratio = opt_in.ratio
    vacancy_symbol = vacancy_symbol_from_swaps(species)

    out_dir = Path(args.out_dir or agent_cfg.get("out_dir", "runs/latest"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_input_manifest(out_dir, opt_in.manifest)
    xyz_writer = SwapStructureWriter(out_dir, species)

    max_iters = int(args.max_iters or agent_cfg.get("max_iters", 20))
    max_swaps = int(agent_cfg.get("max_swaps_per_step", 4))
    seed = int(args.seed if args.seed is not None else agent_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    print_loading_info(opt_in, cfg)
    
    # Swap sites use placeholder element; framework atoms stay fixed.
    template_sites, framework = split_lattice_and_framework(reference, lattice_element)
    n_sites = len(template_sites)
    counts = resolve_counts(n_sites, species, ratio)
    print(f"Swap sites ({lattice_element} placeholder): {n_sites}")
    print(f"Species: {species}")
    print(f"Target counts: {counts}")

    # Setup MatCalc relaxation.
    relax_kwargs = deepcopy(relax)
    relax_kwargs["traj_file"] = str(out_dir / "relaxation.xyz")
    optimizer = setup_optimizer(**relax_kwargs)

    # Create initial occupancy and relax.
    occupancy = initial_occupancy(n_sites, species, ratio, rng)
    print(f"Initial occupancy: {''.join(occupancy)}")
    energy, _results, structure = evaluate_occupancy(
        occupancy, template_sites, framework, optimizer, vacancy_symbol=vacancy_symbol or "X"
    )
    write(out_dir / "initial.xyz", structure)
    xyz_writer.write(structure, label="step_000_initial", occupancy=occupancy, energy=energy)
    print(f"Initial energy: {energy:.8f} eV")

    # Initialize optimizer state.
    state = OptimizerState(occupancy=occupancy, energy=energy)
    state.seen.add(state.occupancy_key())
    state.history.append(
        StepRecord(step=0, energy=energy, occupancy=occupancy.copy(), swaps=[], rationale="initial")
    )

    # Initialize best state.
    best_occ = occupancy.copy()
    best_energy = energy
    best_structure = structure

    # Initialize agent.
    skill_path = Path(args.skill or agent_cfg.get("skill_path", DEFAULT_SKILL))
    if not skill_path.is_absolute():
        skill_path = ROOT / skill_path
    skill_text = skill_path.read_text(encoding="utf-8")
    backend = str(args.agent or agent_cfg.get("backend", "openai"))
    openai_key = (
        args.api_key
        or agent_cfg.get("api_key")
        or cfg.get("openai_api")
    )
    cursor_key = (
        getattr(args, "cursor_api_key", None)
        or agent_cfg.get("cursor_api_key")
        or cfg.get("cursor_api_key")
        or cfg.get("cusor_api")  # legacy typo in config.toml
    )
    agent = SkillAgent(
        skill_text=skill_text,
        backend=backend,
        model=args.model or agent_cfg.get("model"),
        base_url=args.base_url or agent_cfg.get("base_url"),
        api_key=openai_key,
        cursor_api_key=cursor_key,
        api_key_env=str(agent_cfg.get("api_key_env", DEFAULT_OPENAI_API_KEY_ENV)),
        temperature=float(agent_cfg.get("temperature", 0.2)),
    )
    print(f"Agent backend: {backend}  model: {agent.model}")
    if backend == "openai":
        print(f"OpenAI-compatible endpoint: {agent.base_url}")


    # Main optimization loop.
    try:
        for step in range(1, max_iters + 1):
            print(f"\n=== Step {step}/{max_iters} ===")
            proposal = agent.propose(
                state,
                species,
                ratio,
                counts,
                lattice_element,
                vacancy_symbol,
                max_swaps,
            )
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
            best_key = state.occupancy_key(best_occ)
            # Restoring best is required after an uphill move; do not re-relax.
            if key == best_key and state.occupancy_key() != best_key:
                state.occupancy = best_occ.copy()
                state.energy = best_energy
                state.last_rejects.clear()
                print(
                    f"Restored best occupancy (already evaluated; "
                    f"E={best_energy:.8f} eV, no re-relax)."
                )
                continue
            if key in state.seen:
                msg = (
                    f"step {step}: swaps={swaps} already seen occ={''.join(new_occ)}"
                )
                state.last_rejects.append(msg)
                print("Rejected: duplicate occupancy.")
                continue

            # Relax full structure and score with MatCalc.
            energy, _results, structure = evaluate_occupancy(
                new_occ, template_sites, framework, optimizer, vacancy_symbol=vacancy_symbol or "X"
            )
            state.seen.add(key)
            state.occupancy = new_occ
            state.energy = energy
            state.last_rejects.clear()
            state.history.append(
                StepRecord(
                    step=step,
                    energy=energy,
                    occupancy=new_occ.copy(),
                    swaps=swaps,
                    rationale=rationale,
                )
            )
            xyz_writer.write(
                structure,
                label=f"step_{step:03d}",
                occupancy=new_occ,
                energy=energy,
            )
            print(f"Energy: {energy:.8f} eV  (best {best_energy:.8f} eV)")

            if energy < best_energy:
                best_energy = energy
                best_occ = new_occ.copy()
                best_structure = structure
                write_atoms_xyz(out_dir / "best.xyz", best_structure)
                xyz_writer.write(
                    best_structure,
                    label="best",
                    occupancy=best_occ,
                    energy=best_energy,
                )
                print("New best configuration.")
    finally:
        agent.close()

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
    write_atoms_xyz(out_dir / "best.xyz", best_structure)
    xyz_writer.write(best_structure, label="best_final", occupancy=best_occ, energy=best_energy)
    print(f"\nDone. Best energy {best_energy:.8f} eV")
    print(f"Structure trajectory: {xyz_writer.trajectory}")
    print(f"Results written to {out_dir}")
    return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic site optimization (configurable species swaps + MatCalc relax)."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.toml")
    p.add_argument(
        "--skill",
        type=Path,
        default=None,
        help="Path to Skill.md (default: ./Skill.md or agent_params.skill_path)",
    )
    p.add_argument(
        "--agent",
        choices=AGENT_BACKENDS,
        default=None,
        help="Swap proposer backend: openai (Hermes/Ollama/vLLM/OpenAI), cursor, or random",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id (e.g. hermes-3, nousresearch/hermes-3-llama-3.1-70b, composer-2.5)",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL or api.openai.com/v1)",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key override (default: agent_params.api_key or OPENAI_API_KEY)",
    )
    p.add_argument(
        "--cursor-api-key",
        type=str,
        default=None,
        help="Cursor API key override (default: agent_params.cursor_api_key or CURSOR_API_KEY)",
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
    run_agent(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
