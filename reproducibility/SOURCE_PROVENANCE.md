# Source provenance and scope

The release branch was prepared from GitHub `main` at `8411aee47026d0110fe3ebcb787919e31e222a83` in a **fresh checkout**. The archives under the historical `Runs` and `Agenticoptimizer_old` directories were read-only inputs; no archived result was rerun or rewritten.

This package then dropped NaSbSe2, leaving 14 systems, and moved the search code into `methods/`, the one-arm wrappers into `campaign/`, and the 280-arm runner into `reproducibility/`. Docstrings and comments were added in that pass. Files edited for those changes are not byte-identical to the archive.

## What still matches the archive

- The 14 CIF SHA-256 values in `all14_cif_sha256.json` are the archived digests for those systems. NaSbSe2 and its CIF files are not in this package.
- `enum_configs/*.toml` for the 14 systems keep the archived fields. Their `cif_path` values are repository-relative (`cif_files/…`) rather than machine-specific absolute paths.
- Twelve distinct physical CIF files back the 14 configurations: `Cu3Au` and `Cu3Au_123` share one CIF, and `FeCo` and `FeCo_331` share another. `cif_files/NaFePO4_olivine_new.cif` is an extra example and is not one of the 14.

## What was reorganized

- Search entry points, surrogate policies, `utils.py`, and `prepare_disordered.py` live in `methods/`. `utils.ROOT` is the repository root, one directory above that folder.
- One-arm wrappers and `execute_frozen_python.py` live in `campaign/`. The 280-arm runner, launcher, and campaign Skill live in `reproducibility/`. The runner preflights 14 inputs and 280 output paths. It is not the archived 15-system runner.
- `campaign/traditional_run.py` resolves relative CIF paths against the repository root and checks `all14_cif_sha256.json`.

The archived outcomes/figures are not included. A new independent GPU run, comparison against the archived result files, final license/citation metadata, and input redistribution review are separate release gates.

## Independent source and input digests

The 14-entry manifest is the exact subset (excluding `NaSbSe2`) of the independently archived `Agenticoptimizer_old/reproducibility/all15_cif_sha256.json`, whose SHA-256 is `857913de4e984e23b246182939f917b2fcc00b577745d9948caa19ee0f56f51d`. The 14 packaged config-resolved CIFs match those archived digests. A manifest generated only from the packaged inputs would not establish independent input provenance.

Two packaged policy modules are byte-identical to their `Agentic_optimizer/Runs/` versions at audit: `methods/agentic_policy_ce.py` has SHA-256 `c8e75bfcb083797caab7b2a60e32d3b937304b609f3f74c2c53dc93f6169f19b`, and `methods/agentic_policy_aliegnn_online.py` has SHA-256 `ddb27df9b37dc29629de7c7ba6fe485dcdf37b3306ff5dbdf49fe9e9e27b3077`. The archived 15-system frozen runner has SHA-256 `6582742625334e65f72c1925822d94f344b61b4b18cf2fb73082e27c16014a1b`; the adapted 14-system runner is **not byte-identical**. Nor is `methods/agentic_policy_v2.py` byte-identical to the archived `agentic_policy_v2.py`. Source checks and dry preflight do not establish numerical equivalence; only a fresh GPU campaign and independent comparison can do that.

The post-run exporter accepts explicitly supplied roots for runs and finite-enumeration reference structures; no optimizer entry point imports it or reads its reference data during online selection. The Na–Fe/Ga case study and historical enumerators are not reproduced by this 14-system campaign.
