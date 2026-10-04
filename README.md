# Agenticoptimizer — publication source package

Fixed-composition occupancy searches evaluated with MatCalc/TensorNet. This repository contains the source for three traditional baselines, an optional live-LLM agent, and four **numerical** online-surrogate policies (ExtraTrees, Bayesian ridge, sparse ARD, and a compact ALiEGNN-inspired ensemble). The surrogate methods do **not** query a language model during proposal selection. The best relaxed energy observed in a finite search is not a proof of a global minimum.

## Contents and provenance

- `agentic_optimizer.py`, `Metropolis_MC.py`, `genetic_algo_optimizer.py`, `utils.py`: traditional/LLM entry points and shared evaluation, restored from the historical `Runs/source_code` snapshot via the old local checkout.
- `agentic_policy_v2.py`, `agentic_policy_ce.py`, `agentic_policy_aliegnn_online.py`, `Skill_all15_four_surrogates_five_seeds_v1.md`: the frozen four-surrogate policies. The graph model is **ALiEGNN-inspired**, not standard ALIGNN and not the original pretrained ALiEGNN model.
- `run_all15_four_surrogates_five_seeds_reproduction_v1.py`: a *relocated* 15-system × 4-surrogate × 5-seed campaign runner. Its output and input paths differ from the historical runner. `execute_frozen_python.py` binds execution to captured runner bytes.
- `enum_configs/`: 15 portable configurations; `cif_files/`: the corresponding physical CIF inputs. `reproducibility/all15_cif_sha256.json` records the archived-input SHA-256 digests; [source provenance](reproducibility/SOURCE_PROVENANCE.md) distinguishes byte-identical snapshots from relocation edits. Config CIF paths were changed from machine-specific absolute paths to repository-relative paths; physical CIF bytes were checked against the digests.
- `reproduce_traditional_run.py`: safe dry preflight by default for one fresh random-swap, fixed-rate MC, or GA arm.

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
.venv/bin/python reproduce_traditional_run.py --system NaSbSe2 --method mc --seed 0
# Only after provisioning the GPU/model and checking the destination:
.venv/bin/python reproduce_traditional_run.py --system NaSbSe2 --method mc --seed 0 --execute
```

Allowed methods are `random`, `mc`, `ga`; seeds are 0–4. The wrapper refuses an existing output path. Random-swap and MC are configured for 200 proposal steps following an evaluated initial occupancy; duplicate proposals are not re-relaxed. MC accepts downhill/equal-energy moves and uphill moves with fixed probability 0.3, **not** a temperature-dependent Metropolis rule. GA initializes 20 members and attempts nine offspring generations of 20; 200 relaxations are a target, not a guarantee if novel offspring cannot be generated. Its initial population is not explicitly deduplicated.

`config.example.toml` is a portable starting point for direct use of the legacy entry points. Copy it to `config.toml` and adjust the input and output. Never commit local API keys or a production `config.toml`.

## Four-surrogate campaign

The full runner starts **300 fresh searches** (15 systems × 5 seeds × 4 policies) and can consume substantial GPU time and memory. There is deliberately **no one-arm CLI flag**. Do not run it from a chat gateway or beside another GPU job. Use an external scheduler or a dedicated resource-limited service, with disk/RAM/device checks and a fresh `runs/reproduction_all15_four_surrogates_five_seeds_v1/` destination, then launch from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python execute_frozen_python.py run_all15_four_surrogates_five_seeds_reproduction_v1.py
```

The initial occupancy is step 0; a surrogate arm has a hard limit of 200 **subsequent proposals** and can stop earlier on explicitly recorded finite-policy-space or elite-coverage evidence. Training and selection use only the current system, frozen policy/Skill, and observations from that arm. A failed campaign retains its lock and partial outputs for inspection; do not delete or reuse them blindly. This full GPU campaign has **not** been executed as part of preparing this repository.

## Release status

Before public release, the project owner must choose a license and confirm citation metadata/coauthor credit, CIF redistribution permissions, model-weight access, and the final manuscript wording. The GitHub repository remains private until that review is complete.
