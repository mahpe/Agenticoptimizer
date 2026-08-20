# Agenticoptimizer

Agentic site-occupancy optimizer for atomistic structures. Three optimizers share the same `config.toml` and MatCalc relaxation helpers in `utils.py`:

| Script | Method |
|--------|--------|
| `agentic_optimizer.py` | LLM agent proposes swaps (`Skill.md`) |
| `genetic_algo_optimizer.py` | ASE genetic algorithm |
| `Metropolis_MC.py` | Metropolis Monte Carlo |

## Setup

Use an environment with ASE and MatCalc (e.g. the postdoc env):

```bash
source /home/mahpe/envs/postdoc/bin/activate
cd /home/mahpe/Code/Agenticoptimizer
```

For the LLM agent backend (OpenAI-compatible APIs — Hermes, Ollama, vLLM, OpenRouter, OpenAI):

```bash
export OPENAI_API_KEY="..."
# Optional for local Ollama / vLLM:
# export OPENAI_BASE_URL="http://localhost:11434/v1"

python agentic_optimizer.py --agent openai --model hermes3
```

Optional Cursor SDK backend:

```bash
pip install cursor-sdk
export CURSOR_API_KEY="cursor_..."
python agentic_optimizer.py --agent cursor --model composer-2.5
```

## Disordered input CIFs

Any partially occupied CIF can be expanded to an integer-occupancy supercell before optimization.

### Option A — automatic from disordered CIF (recommended)

Set `cif_path` to your disordered CIF (e.g. `cif_files/AuCu_disordered_fcc.cif`) and keep
`[prep_params].auto_prepare = true`. All three optimizers will:

1. Detect disorder groups and read `atom_swaps` / `ratio` from the CIF
2. Build the **smallest supercell** with integer atom counts at that concentration
3. Feed the prepared structure to agent / GA / MC

Optional overrides in `config.toml`:

```toml
[prep_params]
supercell_size = [2, 2, 2]   # manual supercell (must match concentration)

[swap_params]
ratio = [0.6, 0.4]           # manual concentration (auto-sizes supercell if needed)
atom_swaps = ["Au", "Cu"]
lattice_sites = "Au"
```

### Option B — manual ordered input

For an **ordered** parent CIF (no partial occupancies), disable auto prep and set everything explicitly:

```toml
cif_path = "NaFePO4_olivine_new.cif"
supercell_size = [1, 3, 2]

[prep_params]
auto_prepare = false

[swap_params]
lattice_sites = "Na"
atom_swaps = ["Na", "X"]
ratio = [0.66, 0.34]
```

### Option C — explicit prep step

```bash
# Inspect disorder groups
python prepare_disordered.py --input-cif /path/to/CoFe.cif --list-sublattices --out-dir /tmp/x

# Write prepared.cif + config.toml
python prepare_disordered.py \
  --input-cif /path/to/CoFe.cif \
  --out-dir prepared/CoFe \
  --sublattice-index 0

# Run any optimizer on the prepared config
python genetic_algo_optimizer.py --config prepared/CoFe/config.toml
python Metropolis_MC.py --config prepared/CoFe/config.toml
python agentic_optimizer.py --config prepared/CoFe/config.toml --agent random
```

Prepared outputs:

- `prepared.cif` — supercell with placeholder atoms on swap sites
- `manifest.json` — min cell size, species ratios, sublattice info
- `config.toml` — ready-to-run settings (`auto_prepare = false`, `supercell_size = [1,1,1]`)

## Usage

```bash
# AuCu — auto from disordered CIF (uses config.toml)
python agentic_optimizer.py --agent random --max-iters 2
python Metropolis_MC.py --num-steps 3 --out-dir runs/mc_aucu
python genetic_algo_optimizer.py --num-gens 1 --pop-size 4 --out-dir runs/ga_aucu

# Agent-driven optimization (Skill.md + LLM backend from config)
python agentic_optimizer.py

# Hermes / Ollama / OpenAI-compatible API
python agentic_optimizer.py --agent openai --model hermes3 --base-url http://localhost:11434/v1

# Random-swap baseline (no LLM; useful to test the relax loop)
python agentic_optimizer.py --agent random --max-iters 3

# Genetic algorithm
python genetic_algo_optimizer.py --num-gens 2 --pop-size 5 --out-dir runs/ga_test

# Metropolis Monte Carlo
python Metropolis_MC.py --num-steps 10 --out-dir runs/mc_test

# Custom config / output
python agentic_optimizer.py --config config.toml --out-dir runs/my_run --seed 1 --max-iters 20
```

### CLI options

| Flag | Description |
|------|-------------|
| `--config` | Path to `config.toml` (default: `./config.toml`) |
| `--skill` | Path to `Skill.md` |
| `--agent` | `openai` (default), `cursor`, or `random` |
| `--model` | Model id (e.g. `hermes3`, `nousresearch/hermes-3-llama-3.1-70b`) |
| `--base-url` | OpenAI-compatible API URL (Ollama, vLLM, OpenRouter, …) |
| `--max-iters` | Max agent swap steps |
| `--seed` | RNG seed for the initial occupancy |
| `--out-dir` | Output directory (default: `runs/latest`) |

### Config (`config.toml`)

- **`[prep_params]`** — auto-expand disordered CIFs (`auto_prepare`, `max_cell_size`, `sublattice_index`)
- **`cif_path` / `supercell_size`** — input structure (manual path for ordered systems)
- **`[swap_params]`** — swap-site placeholder (`lattice_sites`), allowed species (`atom_swaps`), and composition (`ratio`; use `X` for vacancies)
- **`[relax_params]`** — MatCalc / ASE relaxation settings
- **`[agent_params]`** — iterations, swaps per step, model, skill path, output dir

### Outputs

Written under `--out-dir` (or `agent_params.out_dir`):

- `best.xyz` — lowest-energy relaxed structure (SinglePointCalculator with energy/forces/stress)
- `atom_swaps.xyz` — trajectory of evaluated configurations (label, occupancy, energy in metadata)
- `summary.json` — energies, occupancies, and swap history
- `relaxation.traj` — ASE trajectory from the latest relax
