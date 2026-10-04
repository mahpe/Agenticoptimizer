#!/usr/bin/env python3
"""Run ONE fresh historical-style random/MC/GA arm, without enumeration DB access.

The archived optimizer entry points remain unchanged; this wrapper merely resolves
per-system config and refuses output collisions. It never reads previous results.
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

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / 'enum_configs'
METHOD_SCRIPT = {'random': 'agentic_optimizer.py', 'mc': 'Metropolis_MC.py', 'ga': 'genetic_algo_optimizer.py'}

def command(system: str, method: str, seed: int, out: Path) -> list[str]:
    config = CONFIG_ROOT / f'{system}.toml'
    if not config.is_file() or system not in {p.stem for p in CONFIG_ROOT.glob('*.toml')}:
        raise ValueError(f'unknown system: {system}')
    cfg = tomllib.loads(config.read_text())
    source = Path(cfg['cif_path'])
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = ROOT / 'reproducibility' / 'all15_cif_sha256.json'
    hashes = json.loads(expected.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[system]:
        raise ValueError(f'historical CIF digest mismatch: {system}')
    common = [sys.executable, str(ROOT / METHOD_SCRIPT[method]), '--config', str(config), '--seed', str(seed), '--out-dir', str(out)]
    if method == 'random': return common + ['--agent', 'random', '--max-iters', '200']
    if method == 'mc': return common + ['--num-steps', '200']
    return common + ['--num-gens', '9', '--pop-size', '20']

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', required=True)
    parser.add_argument('--method', required=True, choices=tuple(METHOD_SCRIPT))
    parser.add_argument('--seed', type=int, required=True, choices=range(5), metavar='{0,1,2,3,4}')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs' / 'reproduction_traditional_v1')
    parser.add_argument('--execute', action='store_true', help='Without this flag, validate and print the command only')
    args = parser.parse_args()
    out = args.output_root.resolve() / args.system / f'{args.method}_seed{args.seed}'
    cmd = command(args.system, args.method, args.seed, out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'refusing to overwrite existing output: {out}')
    print(json.dumps({'mode': 'execute' if args.execute else 'preflight_only', 'command': cmd, 'output': str(out)}, indent=2), flush=True)
    if not args.execute: return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

if __name__ == '__main__':
    raise SystemExit(main())
