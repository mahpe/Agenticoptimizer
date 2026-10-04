# Source provenance and scope

The release branch was prepared from GitHub `main` at `8411aee47026d0110fe3ebcb787919e31e222a83` in a **fresh checkout**. The archives under the historical `Runs` and `Agenticoptimizer_old` directories were read-only inputs; no archived result was rerun or rewritten.

## Byte-identical snapshots

- `agentic_optimizer.py`, `genetic_algo_optimizer.py`, `Metropolis_MC.py`, and `utils.py` are byte-identical to their counterparts in the archived `Runs/source_code/` directory at preparation time. In particular, `agentic_optimizer.py` SHA-256 is `11767c1dfba954c76172e84e12274e438d810afaaddf0c958a44eee3e9928f9c`.
- `Skill_all15_four_surrogates_five_seeds_v1.md` is byte-identical to the frozen archived Skill, SHA-256 `d97d495024f07bbe670f92b345ca5bd8f8d50f57fbdad7c06427788dafb87a62`.
- The 15 named input CIF SHA-256 values are in `all15_cif_sha256.json`; 13 distinct physical CIF files back the configurations. All archived hashes and configurations were checked after copying.
- The surrogate policies and reproduction runner came from the historical `Agenticoptimizer_old` checkout. Only the runner and traditional wrapper received the portability edits below; the surrogate policy files were copied verbatim.

## Intentional relocations

- The 15 TOML `cif_path` values now point to `cif_files/…` relative to the repository root, replacing machine-specific absolute paths. Other TOML fields are unchanged.
- `run_all15_four_surrogates_five_seeds_reproduction_v1.py` resolves a relative `cif_path` against its own source directory and now preflights all 15 inputs and 300 output paths before launch. Its current SHA-256 is `db6842202a6d259560e525e558f44fbaaed0b5e6b3b72d31e68ebb041130d138`, whereas its direct old-checkout source had SHA-256 `7f71fd42f71022db07ae72cce0750094d77f9d120dd014b232ab8656cc6a9960`. It is **not** represented as byte-identical to the archived campaign runner (`6582742625334e65f72c1925822d94f344b61b4b18cf2fb73082e27c16014a1b`). The runner freezes and hashes its own code and input state when launched.
- `reproduce_traditional_run.py` likewise resolves relative CIF paths against the repository root.

The archived outcomes/figures are not included. A new independent GPU run, comparison against the archived result files, final license/citation metadata, and input redistribution review are separate release gates.
