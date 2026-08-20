---
name: atom-site-swap
description: >-
  Propose fixed-composition lattice-site swaps to lower the relaxed energy of a
  configuration. Works for any swap species defined in config.toml (e.g. Fe/Co,
  Li/Mn, Na/vacancy). Use when optimizing site occupancy in Agenticoptimizer.
---

# Atomistic site-swap agent

## Goal

Find a low-energy arrangement of species on **swap sites** at a **fixed composition**.
The allowed species, target ratio, and site count come from the runtime prompt (derived
from `config.toml` / disordered CIF preparation). You choose which sites to exchange.

Do **not** change the count of any swap species. Only permute occupations on swap sites.

## Domain rules

1. **Swap sites** are indexed `0 .. N-1`. Each step you receive the current assignment
   of species to those indices.
2. **Allowed species** are exactly those listed in the runtime prompt (`atom_swaps`).
   Use only those symbols — do not invent elements.
3. If a species is **`X`**, it denotes a **vacancy** (empty site). Vacancies are removed
   before relaxation. Not all systems have vacancies; many are multi-cation swaps only
   (e.g. Fe/Co, Li/Ni).
4. **Composition is fixed.** Target counts for each species are given in the runtime
   prompt. Every accepted move must preserve those counts exactly.
5. **Framework atoms are fixed.** Only swap-site occupations may change.
6. **Lower relaxed total energy is better.** Break unfavorable clustering or try new
   arrangements when progress stalls.

## How to propose a move

Each step the runtime prompt includes:

- allowed species and target counts
- current occupancy (one symbol per swap-site index)
- current energy (eV) and recent history

Respond with **only** a JSON object (no markdown fences, no commentary):

```json
{"swaps": [[i, j], [k, l]], "rationale": "short reason"}
```

- Each `[i, j]` exchanges the species on swap sites `i` and `j`.
- Prefer 1–4 swaps per step unless more is clearly justified.
- Only swap pairs with **different** species (same-species swaps are no-ops).
- Indices must refer to swap sites only and stay in range.

To stop when further improvement seems unlikely:

```json
{"swaps": [], "done": true, "rationale": "why stop"}
```

## Strategy hints

- Early: diversify — break large clusters of the same species.
- Mid: exploit — refine regions that previously lowered energy.
- Late: small local swaps; stop if energy plateaus.
- Never change species counts or repeat a prior occupancy from history.

## What you must not do

- Do not change cell size, framework chemistry, or target composition.
- Do not request calculator/relaxation setting changes mid-run.
- Do not output prose outside the JSON object.
