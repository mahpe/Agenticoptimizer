# Agenticoptimizer

Agentic site-occupancy optimizer for atomistic structures. Like a genetic algorithm over lattice sites, but an agent (guided by `Skill.md`) proposes Na/X swaps instead of GA operators. Structures are relaxed and scored with MatCalc via `Main_scripts.py`.

## Setup

Use an environment with ASE and MatCalc (e.g. the postdoc env):

```bash
source /home/mahpe/envs/postdoc/bin/activate
cd /home/mahpe/Code/Agenticoptimizer
```

For the Cursor agent backend:

```bash
pip install cursor-sdk
export CURSOR_API_KEY="cursor_..."
```

## Usage

```bash
# Agent-driven optimization (reads config.toml + Skill.md)
python run.py --agent cursor

# Random-swap baseline (no LLM; useful to test the relax loop)
python run.py --agent random --max-iters 3

# Custom config / output
python run.py --config config.toml --out-dir runs/my_run --seed 1 --max-iters 20
```

### CLI options

| Flag | Description |
|------|-------------|
| `--config` | Path to `config.toml` (default: `./config.toml`) |
| `--skill` | Path to `Skill.md` |
| `--agent` | `cursor` (Skill.md + Cursor SDK) or `random` |
| `--model` | Cursor model id (default from config: `composer-2.5`) |
| `--max-iters` | Max agent swap steps |
| `--seed` | RNG seed for the initial occupancy |
| `--out-dir` | Output directory (default: `runs/latest`) |

### Config (`config.toml`)

- **`cif_path` / `supercell_size`** — input structure
- **`[swap_params]`** — lattice element, swap species (`Na` / `X`), and composition ratio (`X` = vacancy)
- **`[relax_params]`** — MatCalc / ASE relaxation settings
- **`[agent_params]`** — iterations, swaps per step, model, skill path, output dir

### Outputs

Written under `--out-dir` (or `agent_params.out_dir`):

- `best.cif` — lowest-energy structure found
- `step_XXX.cif` — structure after each accepted step
- `summary.json` — energies, occupancies, and swap history
- `relaxation.traj` — ASE trajectory from the latest relax
