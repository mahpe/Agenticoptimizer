#!/usr/bin/env python3
"""Run ONE fresh ExtraTrees, Bayesian ridge, sparse ARD, or ALiEGNN arm.

Same usage as ``traditional_run.py``: choose a system, one surrogate,
and a seed. Dry preflight is the default. ``--execute-gpu`` runs TensorNet on
CUDA (``cuda:0``, the paper setting); ``--execute-cpu`` runs it on the host CPU.
Execution goes through the captured-byte executor. This wrapper never reads
previous results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'enum_configs'
EXECUTOR = Path(__file__).resolve().parent / 'execute_frozen_python.py'
RUNNER = ROOT / 'reproducibility' / 'run_all14_four_surrogates_five_seeds_reproduction_v1.py'
METHODS = ('extra_trees', 'bayesian_ridge', 'ard', 'aliegnn_online')
OUT_PREFIX = 'agentic_without_unique_database_all14_four_surrogates_five_seeds_v1_'


def output_path(system: str, method: str, seed: int) -> Path:
    """Campaign folder for this arm. The runner writes here regardless of cwd."""
    return (ROOT / 'runs' / 'reproduction_all14_four_surrogates_five_seeds_v1'
            / system / f'{OUT_PREFIX}{method}_seed{seed}')


def command(system: str, method: str, seed: int, device: str = 'gpu') -> list[str]:
    """Build the one-arm campaign command. Does not create the output folder or relax."""
    if method not in METHODS:
        raise ValueError(f'unknown method: {method}')
    if device not in {'gpu', 'cpu'}:
        raise ValueError(f'device must be gpu or cpu, got {device}')
    config = CONFIG_ROOT / f'{system}.toml'
    if not config.is_file() or system not in {p.stem for p in CONFIG_ROOT.glob('*.toml')}:
        raise ValueError(f'unknown system: {system}')
    cfg = tomllib.loads(config.read_text())
    source = Path(cfg['cif_path'])
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(source)
    hashes = json.loads((ROOT / 'reproducibility' / 'all14_cif_sha256.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[system]:
        raise ValueError(f'historical CIF digest mismatch: {system}')
    return [
        sys.executable, str(EXECUTOR), str(RUNNER),
        '--system', system, '--method', method, '--seed', str(seed),
        '--device', device,
    ]


def execution_env(device: str) -> dict[str, str]:
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    env['AGENTIC_DEVICE'] = 'cuda:0' if device == 'gpu' else 'cpu'
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', required=True)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--seed', type=int, required=True, choices=range(5), metavar='{0,1,2,3,4}')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute-gpu', action='store_true',
                       help='Start the run with TensorNet on CUDA (cuda:0)')
    group.add_argument('--execute-cpu', action='store_true',
                       help='Start the run with TensorNet on CPU')
    args = parser.parse_args()
    if args.execute_gpu:
        mode, device = 'execute_gpu', 'gpu'
    elif args.execute_cpu:
        mode, device = 'execute_cpu', 'cpu'
    else:
        mode, device = 'preflight_only', 'gpu'
    out = output_path(args.system, args.method, args.seed)
    cmd = command(args.system, args.method, args.seed, device=device)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'refusing to overwrite existing output: {out}')
    print(json.dumps({
        'mode': mode,
        'device': device,
        'command': cmd,
        'output': str(out),
    }, indent=2), flush=True)
    if mode == 'preflight_only':
        return 0
    return subprocess.call(cmd, cwd=ROOT, env=execution_env(device))


if __name__ == '__main__':
    raise SystemExit(main())
