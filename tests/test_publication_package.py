"""Source-package checks that do not launch a GPU campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = (
    'AlReGe', 'Ba2ScTaO6', 'CoPt', 'Cu3Au', 'Cu3Au_123',
    'FeCo', 'FeCo_331', 'FePt', 'InCuTe2', 'LiYS2',
    'Ni2SbTe2', 'ThMn2Co3', 'TiCuNi', 'V2ZnO4',
)


def test_portable_configs_bind_all_historical_cif_bytes():
    expected = json.loads((ROOT / 'reproducibility/all14_cif_sha256.json').read_text())
    assert set(expected) == set(SYSTEMS)
    assert {p.stem for p in (ROOT / 'enum_configs').glob('*.toml')} == set(SYSTEMS)
    for system in SYSTEMS:
        config = tomllib.loads((ROOT / 'enum_configs' / f'{system}.toml').read_text())
        path = Path(config['cif_path'])
        assert not path.is_absolute()
        source = (ROOT / path).resolve()
        assert source.is_relative_to(ROOT / 'cif_files')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected[system]
        assert config['ga_params']['pop_size'] == 20
        assert config['ga_params']['num_gens'] == 9
        assert config['mc_params']['acceptance_rate'] == 0.3


@pytest.mark.parametrize('method', ['random', 'mc', 'ga'])
def test_baseline_preflight_is_fresh_and_does_not_evaluate(tmp_path, method):
    from traditional_run import command

    out = tmp_path / 'new-output'
    argv = command('FePt', method, 0, out)
    assert '--config' in argv and str(out) in argv
    assert not out.exists()
    if method == 'mc':
        assert '--num-steps' in argv
    else:
        assert '--num-steps' not in argv


@pytest.mark.parametrize('method', ['extra_trees', 'bayesian_ridge', 'ard', 'aliegnn_online'])
def test_agentic_preflight_is_fresh_and_does_not_evaluate(method):
    from agentic_run import command, output_path

    argv = command('FePt', method, 0)
    assert argv[1].endswith('execute_frozen_python.py')
    assert argv[2].endswith('run_all14_four_surrogates_five_seeds_reproduction_v1.py')
    assert argv[argv.index('--system') + 1] == 'FePt'
    assert argv[argv.index('--method') + 1] == method
    assert argv[argv.index('--seed') + 1] == '0'
    assert argv[argv.index('--device') + 1] == 'gpu'
    assert command('FePt', method, 0, device='cpu')[-1] == 'cpu'
    assert output_path('FePt', method, 0).name.endswith(f'{method}_seed0')


def test_example_config_requires_no_host_paths():
    cfg = tomllib.loads((ROOT / 'config.example.toml').read_text())
    assert (ROOT / cfg['cif_path']).is_file()
    assert cfg['agent_params']['backend'] == 'random'


def test_campaign_preflight_rejects_any_arm_collision_before_writing(tmp_path, monkeypatch):
    import importlib.util

    runner = ROOT / 'reproducibility' / 'run_all14_four_surrogates_five_seeds_reproduction_v1.py'
    monkeypatch.setenv('HERMES_EXECUTED_RUNNER_SHA256', hashlib.sha256(runner.read_bytes()).hexdigest())
    spec = importlib.util.spec_from_file_location('publication_campaign', runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'RUN_ROOT', tmp_path / 'campaign')
    monkeypatch.setattr(module, 'CAMPAIGN_LOCK', module.RUN_ROOT / '.all14_four_surrogates_five_seeds_v1.lock')
    monkeypatch.setattr(module, 'BATCH_STATUS', module.RUN_ROOT / '_all14_four_surrogates_five_seeds_v1_status.json')
    # Pick the last arm so the check covers the full 280-arm destination set.
    collision = module.RUN_ROOT / module.SYSTEMS[-1] / f'{module.OUT_PREFIX}{module.SURROGATES[-1]}_seed{module.SEEDS[-1]}'
    collision.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='existing arm outputs'):
        module.preflight_campaign()
    assert not module.CAMPAIGN_LOCK.exists()
    assert not module.BATCH_STATUS.exists()
    assert len(list(module.RUN_ROOT.rglob('*'))) == 2  # system directory and collision only


def test_one_arm_preflight_rejects_only_that_arm(tmp_path, monkeypatch):
    import importlib.util

    runner = ROOT / 'reproducibility' / 'run_all14_four_surrogates_five_seeds_reproduction_v1.py'
    monkeypatch.setenv('HERMES_EXECUTED_RUNNER_SHA256', hashlib.sha256(runner.read_bytes()).hexdigest())
    spec = importlib.util.spec_from_file_location('publication_campaign_one_arm', runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'RUN_ROOT', tmp_path / 'campaign')
    other = module.arm_output_path('FePt', 'extra_trees', 0)
    other.mkdir(parents=True)
    module.preflight_one_arm('FePt', 'bayesian_ridge', 0)
    collision = module.arm_output_path('FePt', 'bayesian_ridge', 0)
    collision.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='existing arm output'):
        module.preflight_one_arm('FePt', 'bayesian_ridge', 0)
