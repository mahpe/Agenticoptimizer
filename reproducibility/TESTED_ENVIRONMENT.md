# Tested source/preflight environment (not a model or dependency lock)

The checks for this 14-system source snapshot were run in a pre-existing environment, **not** a fresh installation of this checkout. The following versions were observed from Python package metadata on the validation host:

| Component | Observed version |
| --- | --- |
| Python | 3.11.15 |
| Linux | aarch64; glibc 2.39; kernel 7.0.0-1019-nvidia |
| PyTorch | 2.13.0+cu130; CUDA build 13.0; `torch.cuda.is_available() == True` |
| numpy | 2.4.6 |
| ase | 3.29.0 |
| ase-ga | 1.0.3 |
| matcalc | 0.5.1 |
| pymatgen | 2026.5.4 |
| scikit-learn | 1.9.0 |
| packaging | 26.2 |
| httpx | 0.28.1 |
| pytest | 9.1.1 |

`requirements.txt` describes the intended direct dependencies, not a transitive lock. Platform-specific CUDA/PyTorch wheels and MatCalc model weights must be provisioned independently and recorded for a numerical reproduction. A fresh-environment install, calculator initialization on the target GPU, one-arm real relaxation, and full-campaign output comparison have **not** been verified by the source tests or dry preflight. Do not infer numerical equivalence merely from matching library names or version numbers.

Validation commands after installing the project dependencies into the checkout's own `.venv`:

```bash
.venv/bin/python -m pytest -q tests
bash reproducibility/launch_all14_four_surrogates_five_seeds_v1.sh --preflight
.venv/bin/python campaign/traditional_run.py --system FePt --method mc --seed 0
```

The last command is a dry command print, not a GPU run. The launcher without `--preflight` requires user-systemd and starts resource-contained GPU work; it must only be invoked when that full resource and output commitment is intended.
