# Source provenance and scope

The release branch was prepared from GitHub `main` at `8411aee47026d0110fe3ebcb787919e31e222a83` in a **fresh checkout**. The archives under the historical `Runs` and `Agenticoptimizer_old` directories were read-only inputs; no archived result was rerun or rewritten.

This package then dropped NaSbSe2, leaving 14 systems, and moved the search code into `methods/` and the campaign scripts into `campaign/`. Docstrings and comments were added in that pass. Files edited for those changes are not byte-identical to the archive.

## What still matches the archive

- The 14 CIF SHA-256 values in `all14_cif_sha256.json` are the archived digests for those systems. NaSbSe2 and its CIF files are not in this package.
- `enum_configs/*.toml` for the 14 systems keep the archived fields. Their `cif_path` values are repository-relative (`cif_files/…`) rather than machine-specific absolute paths.
- Twelve distinct physical CIF files back the 14 configurations: `Cu3Au` and `Cu3Au_123` share one CIF, and `FeCo` and `FeCo_331` share another. `cif_files/NaFePO4_olivine_new.cif` is an extra example and is not one of the 14.

## What was reorganized

- Search entry points, surrogate policies, `utils.py`, and `prepare_disordered.py` live in `methods/`. `utils.ROOT` is the repository root, one directory above that folder.
- The campaign runner, launcher, captured-byte executor, traditional-run wrapper, and campaign Skill live in `campaign/`. The runner preflights 14 inputs and 280 output paths. It is not the archived 15-system runner.
- `campaign/reproduce_traditional_run.py` resolves relative CIF paths against the repository root and checks `all14_cif_sha256.json`.

The archived outcomes/figures are not included. A new independent GPU run, comparison against the archived result files, final license/citation metadata, and input redistribution review are separate release gates.
