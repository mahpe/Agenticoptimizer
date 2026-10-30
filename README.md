# Agenticoptimizer

Source code for **Agentic Global Optimization of Site-Disordered Crystalline Materials**.

Site-disordered crystals (several elements sharing one Wyckoff site) cannot be simulated atomistically until an explicit ordering is chosen. The number of composition-preserving orderings grows combinatorially with supercell size, and each candidate needs an expensive structural relaxation. This package treats that search as a fixed-composition global optimization problem: find the occupancy of the swap sites that minimizes relaxed total energy.

A surrogate fitted only to occupancies already relaxed in the current run selects the next candidate by balancing exploitation, uncertainty, and novelty. 

## How to run the seven methods

Pick a system name from `enum_configs/` (for example `FePt`). All seven methods relax candidates with TensorNet via MatCalc, keep the composition fixed, and skip occupancies that were already evaluated. Work from the repository root after installing the environment below.

**Traditional methods** — one system, one method, one seed at a time:

```bash
python campaign/traditional_run.py --system FePt --method random --seed 0
python campaign/traditional_run.py --system FePt --method mc --seed 0
python campaign/traditional_run.py --system FePt --method ga --seed 0
```

Without an execute flag the command is printed and nothing is evaluated. To start a run, add `--execute-gpu` (TensorNet on `cuda:0`, the paper setting) or `--execute-cpu` (TensorNet on the host CPU; slower and not bitwise identical to CUDA):

```bash
python campaign/traditional_run.py --system FePt --method mc --seed 0 --execute-gpu
python campaign/traditional_run.py --system FePt --method mc --seed 0 --execute-cpu
```


| `--method` | What it does                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------- |
| `random`   | Random unlike-site swaps (1–4 pairs) from the current occupancy                                 |
| `mc`       | Same swap proposals; downhill moves always accepted, uphill moves accepted with probability 0.3 |
| `ga`       | Genetic algorithm: population 20, up to nine generations                                        |


`--seed` is 0–4. Output goes under `runs/reproduction_traditional_v1/<system>/<method>_seed<seed>/`. The wrapper will not overwrite an existing folder.

**Agentic surrogates** — one system, one method, one seed, same pattern as the traditional wrapper:

```bash
python campaign/agentic_run.py --system FePt --method extra_trees --seed 0
python campaign/agentic_run.py --system FePt --method bayesian_ridge --seed 0
python campaign/agentic_run.py --system FePt --method ard --seed 0
python campaign/agentic_run.py --system FePt --method aliegnn_online --seed 0
```

Without an execute flag the command is printed and nothing is evaluated. To start a run, add `--execute-gpu` or `--execute-cpu`:

```bash
python campaign/agentic_run.py --system FePt --method bayesian_ridge --seed 0 --execute-gpu
python campaign/agentic_run.py --system FePt --method bayesian_ridge --seed 0 --execute-cpu
```


| `--method`       | Energy model used to pick the next occupancy                  |
| ---------------- | ------------------------------------------------------------- |
| `extra_trees`    | Extra-trees ensemble; uncertainty is disagreement among trees |
| `bayesian_ridge` | Linear fit of cluster correlations (recommended default)      |
| `ard`            | Same linear descriptor, with unused features pruned           |
| `aliegnn_online` | Three small graph networks trained only on this run           |


Output goes under `runs/reproduction_all14_four_surrogates_five_seeds_v1/<system>/agentic_without_unique_database_all14_four_surrogates_five_seeds_v1_<method>_seed<seed>/`. The wrapper will not overwrite an existing folder.

## Reproduce the results from the paper

**Agenmtic run**

To run all 14 systems × 4 policies × 5 seeds in one job (280 searches):

```bash
bash reproducibility/launch_all14_four_surrogates_five_seeds_v1.sh --preflight
bash reproducibility/launch_all14_four_surrogates_five_seeds_v1.sh
```

The three traditional methods must be run separately (14 × 3 × 5 = 210 arms) with `campaign/traditional_run.py`. After every summary exists, point the exporter at a versioned finite-enumeration reference (`<system>/optimized_all.extxyz` for each of the 14 cases):

```bash
.venv/bin/python analysis/export_first_hits.py \
  --traditional-root runs/reproduction_traditional_v1 \
  --surrogate-root runs/reproduction_all14_four_surrogates_five_seeds_v1 \
  --reference-root /path/to/finite-reference-structures \
  --runner-source reproducibility/run_all14_four_surrogates_five_seeds_reproduction_v1.py \
  --output first_hits_all14_7methods.csv
```

The CSV is semicolon-delimited. The default hit tolerance is 2×10⁻⁵ eV. GA hits are reported as a generation evaluation interval, not an exact first-evaluation index. First-hit step is retrospective; runs that continued after the hit still spent that extra compute.

`plot_all_convergence_runs.ipynb` is an exploratory notebook against the original `Runs/` archive, not the portable figure path.

To search a structure that is not in the table, copy `config.example.toml` to `config.toml`, set `cif_path`, and call `methods/agentic_optimizer.py`, `methods/Metropolis_MC.py`, or `methods/genetic_algo_optimizer.py` directly. Do not commit `config.toml`.

Available `--system` names:


| Config name | Formula in the paper | Supercell |
| ----------- | -------------------- | --------- |
| `AlReGe`    | Al₁.₂₅Ge₀.₇₅Re       | 2×2×1     |
| `Ba2ScTaO6` | BaSc₀.₅Ta₀.₅O₃       | 2×2×4     |
| `CoPt`      | CoPt                 | 1×2×2     |
| `Cu3Au`     | Au₀.₈Cu₀.₂           | 1×1×5     |
| `Cu3Au_123` | Au₀.₈Cu₀.₂           | 1×2×3     |
| `FeCo`      | FeCo                 | 2×2×2     |
| `FeCo_331`  | FeCo                 | 3×3×1     |
| `FePt`      | Fe₃Pt                | 1×1×5     |
| `InCuTe2`   | CuInTe₂              | 1×2×2     |
| `LiYS2`     | LiYS₂                | 1×2×2     |
| `Ni2SbTe2`  | Ni₂SbTe₂             | 2×2×2     |
| `ThMn2Co3`  | ThCo₃Mn₂             | 1×1×3     |
| `TiCuNi`    | TiCuNi               | 1×2×2     |
| `V2ZnO4`    | Zn₂VO₄               | 1×1×1     |


## Install

Python 3.11+ and a CUDA stack that can load TensorNet are required for energy evaluations. Pinned versions in `requirements.txt` match the [source-test environment](reproducibility/TESTED_ENVIRONMENT.md); they are not a full CUDA/model lock. Tests and dry preflight do not start a GPU relaxation:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

## Citation

Martin Hoffmann Petersen, Zekun Ren, and Kedar Hippalgaonkar, *Agentic Global Optimization of Site-Disordered Crystalline Materials*.

```
@article{petersen2026agentic,
  title   = {Agentic Global Optimization of Site-Disordered Crystalline Materials},
  author  = {Petersen, Martin Hoffmann and Ren, Zekun and Hippalgaonkar, Kedar},
  year    = {2026}
}
```

