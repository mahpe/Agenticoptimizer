---
name: atom-site-swap
description: >-
  Propose Na/X (or configured) lattice-site swaps to lower the relaxed energy
  of a fixed-composition configuration. Use when optimizing site occupancy in
  Agenticoptimizer, reading energy feedback, or choosing which atoms to exchange.
---

# Atomistic site-swap agent

## Goal

Find a low-energy arrangement of species on the **lattice sites** at a **fixed composition** (ratios from `config.toml`). This replaces genetic-algorithm operators: **you** choose which sites to swap.

Do **not** change the number of each species. Only permute occupations on lattice sites.

## Domain rules

1. Lattice sites come from `swap_params.lattice_sites` (e.g. all Na positions in the supercell).
2. Allowed species are `swap_params.atom_swaps` (e.g. `Na` and `X`).
3. `X` means a **vacancy**. It is removed before relaxation; do not invent other elements.
4. Composition must stay at `swap_params.ratio` (e.g. `[0.66, 0.34]` → ~66% first species, ~34% second).
5. Framework atoms (everything that is not a lattice site) are fixed — never swap them.
6. Lower relaxed total energy is better. Prefer swaps that break unfavorable clustering or improve mixing when energy stalls.

## How to propose a move

Each step you receive:

- current occupancy as a list of symbols, one per lattice site index `0 .. N-1`
- current energy (eV) and recent history
- counts of each species (must be preserved)

Respond with **only** a JSON object (no markdown fences, no commentary):

```json
{"swaps": [[i, j], [k, l]], "rationale": "short reason"}
```

- Each `[i, j]` exchanges the species on sites `i` and `j`.
- Prefer 1–4 swaps per step; more is allowed if justified.
- Only swap pairs that currently have **different** species (otherwise the move is a no-op).
- Site indices must be in range and refer to lattice sites only.

If you believe the current configuration is optimal enough to stop:

```json
{"swaps": [], "done": true, "rationale": "why stop"}
```

## Strategy hints

- Early: diversify — break large same-species clusters; sample different local arrangements.
- Mid: exploit — refine neighborhoods that previously lowered energy.
- Late: small local swaps only; stop when energy no longer improves for several steps.
- Never propose a move that changes species counts.
- Avoid repeating an occupancy that already appears in the history.

## What you must not do

- Do not change cell size, framework chemistry, or the target ratio.
- Do not request DFT settings changes mid-run; relaxation params come from config.
- Do not output prose outside the JSON object.
