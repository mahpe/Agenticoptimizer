# Agenticoptimizer — publication source package

Fixed-composition occupancy searches evaluated with MatCalc/TensorNet. This repository contains the source for three traditional baselines, an optional live-LLM agent, and four **numerical** online-surrogate policies (ExtraTrees, Bayesian ridge, sparse ARD, and a compact ALiEGNN-inspired ensemble). The surrogate methods do **not** query a language model during proposal selection. The best relaxed energy observed in a finite search is not a proof of a global minimum.

The publication set is **14 systems**. NaSbSe2 is not included.

## Layout

- `methods/`: search code.
  - `agentic_optimizer.py`, `Metropolis_MC.py`, `genetic_algo_optimizer.py`: random-swap, fixed-rate Monte Carlo, and genetic algorithm.
  - `agentic_policy_v2.py`, `agentic_policy_ce.py`, `agentic_policy_aliegnn_online.py`: ExtraTrees, linear cluster-expansion (Bayesian ridge and sparse ARD), and the compact ALiEGNN-inspired ensemble. The graph model is **ALiEGNN-inspired**, not standard ALIGNN and not the original pretrained ALiEGNN model.
  - `utils.py`: structure preparation, relaxation, and occupancy helpers shared by the entry points.
  - `prepare_disordered.py`: build an integer-occupancy supercell and a matching `config.toml`.
- `campaign/`: the 14-system reproduction campaign.
  - `run_all14_four_surrogates_five_seeds_reproduction_v1.py`: 14 systems × 4 surrogates × 5 seeds. `execute_frozen_python.py` binds execution to one captured copy of that runner.
  - `launch_all14_four_surrogates_five_seeds_v1.sh`: resource-limited launcher. `--preflight` does not start an evaluator.
  - `reproduce_traditional_run.py`: dry preflight by default for one random-swap, Monte Carlo, or genetic-algorithm arm.
  - `Skill_all14_four_surrogates_five_seeds_v1.md`: frozen campaign contract for the four numerical surrogates.
- `Skill.md`: instructions for the optional live-LLM agent. The surrogate campaign does not use it to choose moves.
- `enum_configs/`: one portable TOML file per system. `cif_files/`: the CIF inputs those configs name, plus `NaFePO4_olivine_new.cif` as a separate example that is not one of the 14. `reproducibility/all14_cif_sha256.json` records the SHA-256 digest of each campaign CIF. [Source provenance](reproducibility/SOURCE_PROVENANCE.md) records what was carried over from the earlier 15-system archive.

**Not included:** archived outcomes, trajectories, enumeration databases, pretrained potential weights, secrets, and the unpublished manuscript draft. None of these may be consulted by the surrogate selection policy. Original outputs remain in the external `Runs/` archive and are not overwritten by this package. No publication results or figures are represented as reproduced by merely releasing this source.

## Install and verify

Use Python 3.11+ with an appropriate CUDA environment for TensorNet evaluation. The pinned package versions in `requirements.txt` match the development environment; a compatible NVIDIA driver/CUDA runtime and model download/access are additional requirements. Installation and source-level testing do not require a GPU evaluation:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

The repository does not include a CUDA/model deployment recipe or claim bitwise identity of a fresh GPU run. Numerical reproducibility additionally depends on the loaded model state, package/CUDA versions, hardware, and relaxation settings. Each surrogate run records source, input, and model-state hashes in its output.

## Fresh single-arm baseline preflight

Run commands from the repository root. The first command checks a named system's input hash and prints the would-be command **without** starting MatCalc or creating outputs:

```bash
.venv/bin/python campaign/reproduce_traditional_run.py --system FePt --method mc --seed 0
# Only after provisioning the GPU/model and checking the destination:
.venv/bin/python campaign/reproduce_traditional_run.py --system FePt --method mc --seed 0 --execute
```

Allowed methods are `random`, `mc`, `ga`; seeds are 0–4. The wrapper refuses an existing output path. Random-swap and MC are configured for 200 proposal steps following an evaluated initial occupancy; duplicate proposals are not re-relaxed. MC accepts downhill/equal-energy moves and uphill moves with fixed probability 0.3, **not** a temperature-dependent Metropolis rule. GA initializes 20 members and attempts nine offspring generations of 20; 200 relaxations are a target, not a guarantee if novel offspring cannot be generated. Its initial population is not explicitly deduplicated.

`config.example.toml` is a portable starting point for direct use of the legacy entry points (`methods/agentic_optimizer.py`, `methods/Metropolis_MC.py`, `methods/genetic_algo_optimizer.py`). Copy it to `config.toml` and adjust the input and output. Never commit local API keys or a production `config.toml`.

## Four-surrogate campaign

The full runner starts **280 fresh searches** (14 systems × 5 seeds × 4 policies) and can consume substantial GPU time and memory. There is deliberately **no one-arm CLI flag**. The **only supported launch path** is the resource-contained systemd user-unit launcher after a dedicated GPU/process review and independent GO audit. Its dry preflight checks all 14 CIF digests, compositions, graph/jump bounds, and 280 absent output paths without starting an evaluator:

```bash
bash campaign/launch_all14_four_surrogates_five_seeds_v1.sh --preflight
# After a separate GPU/resource review and approval:
bash campaign/launch_all14_four_surrogates_five_seeds_v1.sh
```

The launcher uses this checkout's `.venv`, sets `MemoryHigh=16G`, `MemoryMax=24G`, `MemorySwapMax=4G`, `TasksMax=128`, `CPUQuota=1600%`, `OOMPolicy=kill`, and `RuntimeMaxSec=7d`, and then invokes the captured-byte executor. The runner repeats preflight before creating a campaign lock. The initial occupancy is step 0; a surrogate arm has a hard limit of 200 **subsequent proposals** and can stop earlier on explicitly recorded finite-policy-space or elite-coverage evidence. Training and selection use only the current system, frozen policy/Skill, and observations from that arm. A failed campaign retains its lock and partial outputs for inspection; do not delete or reuse them blindly. This full GPU campaign has **not** been executed as part of preparing this repository.

**Historical wording caveats:** The frozen Skill says ExtraTrees uses four workers, whereas `methods/agentic_policy_v2.py` passes `n_jobs=-1` to scikit-learn. The live-LLM prompt in `methods/agentic_optimizer.py` also mentions unsuccessful-attempt/motif-diversity requirements absent from the revised `Skill.md`. Do not present either mismatch as an enforced policy until code and Skill are deliberately versioned together.

## Release status

Before public release, the project owner must choose a license and confirm citation metadata/coauthor credit, CIF redistribution permissions, model-weight access, and the final manuscript wording. The GitHub repository remains private until that review is complete.
