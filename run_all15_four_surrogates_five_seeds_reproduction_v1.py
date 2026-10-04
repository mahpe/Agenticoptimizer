#!/usr/bin/env python3
"""Strict-isolation four-surrogate campaign on 15 configured systems, five seeds each.

Production selection reads only the frozen Skill/policy, current system input,
and observations from this run. It may stop earlier on frozen evidence but must
never evaluate more than 200 proposals after the initial step-0 structure.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
import time
import tomllib
import types
from pathlib import Path

import numpy as np
import torch
from ase.io import write

PUDA_ROOT = Path(__file__).resolve().parent
MAHPE_ROOT = PUDA_ROOT
BASE_POLICY = PUDA_ROOT/'agentic_policy_v2.py'
CE_POLICY = PUDA_ROOT/'agentic_policy_ce.py'
ALIE_POLICY = PUDA_ROOT/'agentic_policy_aliegnn_online.py'
UTILS = PUDA_ROOT/'utils.py'
SKILL = PUDA_ROOT/'Skill_all15_four_surrogates_five_seeds_v1.md'
FROZEN_SOURCE_PATHS = {
    'skill_sha256': SKILL,
    'base_policy_sha256': BASE_POLICY,
    'ce_policy_sha256': CE_POLICY,
    'aliegnn_policy_sha256': ALIE_POLICY,
    'utils_sha256': UTILS,
    'runner_sha256': Path(__file__).resolve(),
}
# Capture before importing local modules, then verify immediately after import.
FROZEN_SOURCE_BYTES = {key: path.read_bytes() for key, path in FROZEN_SOURCE_PATHS.items()}
STARTUP_SOURCE_HASHES = {
    key: hashlib.sha256(payload).hexdigest()
    for key, payload in FROZEN_SOURCE_BYTES.items()
}
EXECUTED_RUNNER_SHA256 = os.environ.get('HERMES_EXECUTED_RUNNER_SHA256')
if EXECUTED_RUNNER_SHA256 != STARTUP_SOURCE_HASHES['runner_sha256']:
    raise RuntimeError('runner must be launched through execute_frozen_python.py from one captured byte buffer')


def load_captured_module(name, path, source_key):
    """Execute a local dependency from the exact captured bytes recorded above."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ''
    sys.modules[name] = module
    payload = FROZEN_SOURCE_BYTES[source_key]
    exec(compile(payload, str(path), 'exec'), module.__dict__, module.__dict__)
    return module


sys.path.insert(0, str(PUDA_ROOT))
sys.path.insert(1, str(MAHPE_ROOT))
utils_module = load_captured_module('utils', UTILS, 'utils_sha256')
SwapStructureWriter = utils_module.SwapStructureWriter
evaluate_occupancy = utils_module.evaluate_occupancy
initial_occupancy = utils_module.initial_occupancy
occupancy_counts = utils_module.occupancy_counts
print_loading_info = utils_module.print_loading_info
resolve_counts = utils_module.resolve_counts
resolve_optimization_input = utils_module.resolve_optimization_input
setup_optimizer = utils_module.setup_optimizer
split_lattice_and_framework = utils_module.split_lattice_and_framework
vacancy_symbol_from_swaps = utils_module.vacancy_symbol_from_swaps
write_atoms_xyz = utils_module.write_atoms_xyz
write_input_manifest = utils_module.write_input_manifest
policy_module = load_captured_module('agentic_policy_v2', BASE_POLICY, 'base_policy_sha256')
balanced_jump_neighbors = policy_module.balanced_jump_neighbors
diverse_elite_archive = policy_module.diverse_elite_archive
effective_acquisition = policy_module.effective_acquisition
elite_stop_evidence = policy_module.elite_stop_evidence
jump_space_exhausted = policy_module.jump_space_exhausted
jump_space_size = policy_module.jump_space_size
MAX_EXHAUSTIVE_JUMP_SPACE = policy_module.MAX_EXHAUSTIVE_JUMP_SPACE
single_neighbors_from_parent = policy_module.single_neighbors_from_parent
select_candidate_v2 = policy_module.select_candidate_v2
ce_policy_module = load_captured_module('agentic_policy_ce', CE_POLICY, 'ce_policy_sha256')
select_candidate_ce = ce_policy_module.select_candidate_ce
aliegnn_policy_module = load_captured_module('agentic_policy_aliegnn_online', ALIE_POLICY, 'aliegnn_policy_sha256')
FixedLineGraphGeometry = aliegnn_policy_module.FixedLineGraphGeometry
select_candidate_aliegnn = aliegnn_policy_module.select_candidate_aliegnn
for _key, _path in FROZEN_SOURCE_PATHS.items():
    if hashlib.sha256(_path.read_bytes()).hexdigest() != STARTUP_SOURCE_HASHES[_key]:
        raise ImportError(f'frozen source changed during captured-module load: {_path}')

CONFIG_ROOT = PUDA_ROOT/'enum_configs'
RUN_ROOT = MAHPE_ROOT/'runs'/'reproduction_all15_four_surrogates_five_seeds_v1'
BATCH_STATUS = RUN_ROOT/'_all15_four_surrogates_five_seeds_v1_status.json'
OUT_PREFIX = 'agentic_without_unique_database_all15_four_surrogates_five_seeds_v1_'
SURROGATES = ('extra_trees', 'bayesian_ridge', 'ard', 'aliegnn_online')
MODEL_NAME = 'TensorNet-MatPES-PBE-2025.2'
SEEDS = tuple(range(5))
MAX_PROPOSALS = 200
MAX_ELITES = 4
MIN_ELITE_HAMMING = 4
JUMP_SIZES = (2, 3, 4)
JUMP_QUOTA = 4
JUMP_INTERVAL = 5
ACQUISITIONS = ('lcb', 'greedy', 'uncertainty', 'novelty')
SYSTEMS = [
    'AlReGe','Ba2ScTaO6','CoPt','Cu3Au','Cu3Au_123','FeCo','FeCo_331',
    'FePt','InCuTe2','LiYS2','NaSbSe2','Ni2SbTe2','ThMn2Co3','TiCuNi','V2ZnO4',
]
assert len(SYSTEMS) == 15

def arm_order(system_index, seed):
    rotation = (system_index + seed) % len(SURROGATES)
    return tuple(SURROGATES[(index + rotation) % len(SURROGATES)]
                 for index in range(len(SURROGATES)))

CAMPAIGN_LOCK = RUN_ROOT/'.all15_four_surrogates_five_seeds_v1.lock'


def geometry(template_sites):
    distances = template_sites.get_all_distances(mic=True)
    shells = []
    for value in sorted(float(distances[i, j]) for i in range(len(distances))
                        for j in range(i + 1, len(distances))):
        if not shells or abs(value - shells[-1]) > 1e-3:
            shells.append(value)
    shell_array = np.asarray(shells)
    pair_shell = {(i, j): int(np.argmin(abs(shell_array - distances[i, j])))
                  for i in range(len(distances)) for j in range(i + 1, len(distances))}
    pair_multiplicity = np.bincount(list(pair_shell.values()), minlength=len(shells)).astype(float)
    signatures = []
    for i, j, k in itertools.combinations(range(len(distances)), 3):
        signatures.append(tuple(sorted((pair_shell[tuple(sorted((i, j)))],
                                        pair_shell[tuple(sorted((i, k)))],
                                        pair_shell[tuple(sorted((j, k)))]))))
    triangle_types = sorted(set(signatures))
    triangle_index = {signature: i for i, signature in enumerate(triangle_types)}
    triangle_multiplicity = np.bincount(
        [triangle_index[signature] for signature in signatures],
        minlength=len(triangle_types),
    ).astype(float)
    return (distances, shells, pair_shell, triangle_index,
            pair_multiplicity, triangle_multiplicity)


def cluster_correlation_descriptor(occupancy, minority, n_shells, pair_shell,
                                   triangle_index, pair_multiplicity,
                                   triangle_multiplicity):
    """Finite-template point, distance-bin pair, and triplet correlations."""
    spins = np.asarray([1.0 if value == minority else -1.0 for value in occupancy])
    pairs = np.zeros(n_shells)
    triangles = np.zeros(len(triangle_index))
    for i, j in itertools.combinations(range(len(spins)), 2):
        pairs[pair_shell[(i, j)]] += spins[i] * spins[j]
    for i, j, k in itertools.combinations(range(len(spins)), 3):
        signature = tuple(sorted((pair_shell[tuple(sorted((i, j)))],
                                  pair_shell[tuple(sorted((i, k)))],
                                  pair_shell[tuple(sorted((j, k)))])))
        triangles[triangle_index[signature]] += spins[i] * spins[j] * spins[k]
    pair_correlations = pairs / np.maximum(pair_multiplicity, 1.0)
    triangle_correlations = triangles / np.maximum(triangle_multiplicity, 1.0)
    return np.concatenate(([1.0, float(spins.mean())],
                           pair_correlations, triangle_correlations))


def sha256_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_sha256(atoms):
    """Hash resolved chemistry, coordinates, periodicity, and cell bytes."""
    digest = hashlib.sha256()
    for array in (
        np.asarray(atoms.numbers, dtype='<i8'),
        np.asarray(atoms.positions, dtype='<f8'),
        np.asarray(atoms.cell.array, dtype='<f8'),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ):
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def calculator_manifest(relax):
    calc_name = relax.get('calc_name', MODEL_NAME)
    if calc_name != MODEL_NAME:
        raise ValueError(f'calculator must be {MODEL_NAME}, got {calc_name}')
    versions = {}
    for package in ('matcalc', 'matgl', 'torch', 'ase', 'numpy', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not-installed'
    return {'model_name': MODEL_NAME, 'package_versions': versions}


def calculator_state_sha256(optimizer):
    """Hash the actual loaded potential parameters used by the evaluator."""
    potential = optimizer.calculator.potential
    digest = hashlib.sha256()
    for name, tensor in sorted(potential.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_paths_unchanged(expected_by_path):
    changed = {
        str(path): (expected, sha256_path(path))
        for path, expected in expected_by_path.items()
        if sha256_path(path) != expected
    }
    if changed:
        raise RuntimeError(f'frozen path changed during run: {changed}')


def assert_frozen_sources_unchanged():
    assert_paths_unchanged({
        path: STARTUP_SOURCE_HASHES[key]
        for key, path in FROZEN_SOURCE_PATHS.items()
    })


def assert_calculator_state_unchanged(optimizer, expected_hash):
    actual = calculator_state_sha256(optimizer)
    if actual != expected_hash:
        raise RuntimeError(f'calculator state changed during run: {actual} != {expected_hash}')


def manifest_sha256(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def serialize_jump_mode_counts(jump_mode_counts):
    return [{'parent_fingerprint': list(parent), 'size': size, 'mode': mode, 'count': count}
            for (parent, size, mode), count in sorted(
                jump_mode_counts.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))]


def write_batch(records):
    payload = json.dumps({'updated_at': time.time(), 'systems': records}, indent=2)
    tmp = BATCH_STATUS.with_name(f'{BATCH_STATUS.name}.{os.getpid()}.tmp')
    tmp.write_text(payload)
    tmp.replace(BATCH_STATUS)


def run_system(system, surrogate, seed):
    if surrogate not in SURROGATES:
        raise ValueError(f'unsupported surrogate {surrogate}; choose from {SURROGATES}')
    assert_frozen_sources_unchanged()
    config_path = CONFIG_ROOT/f'{system}.toml'
    out = RUN_ROOT/system/f'{OUT_PREFIX}{surrogate}_seed{seed}'
    summary_path = out/'summary.json'
    source_hashes = dict(STARTUP_SOURCE_HASHES)
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    cfg = tomllib.loads(config_bytes.decode())
    if 'supercell_size' in cfg and isinstance(cfg['supercell_size'], list):
        cfg['supercell_size'] = tuple(cfg['supercell_size'])
    source_path = Path(cfg['cif_path'])
    if not source_path.is_absolute():
        source_path = PUDA_ROOT / source_path
    source_path = source_path.resolve()
    source_bytes = source_path.read_bytes()
    source_cif_hash = hashlib.sha256(source_bytes).hexdigest()
    frozen_input_root = PUDA_ROOT/'.frozen_benchmark_inputs'
    frozen_input_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = frozen_input_root/f'{system}_{source_cif_hash}.cif'
    if snapshot_path.exists() and snapshot_path.read_bytes() != source_bytes:
        raise RuntimeError(f'frozen input snapshot collision: {snapshot_path}')
    if not snapshot_path.exists():
        snapshot_tmp = snapshot_path.with_suffix('.cif.tmp')
        snapshot_tmp.write_bytes(source_bytes)
        snapshot_tmp.replace(snapshot_path)
    expected_input_paths = {
        config_path: config_hash,
        source_path: source_cif_hash,
        snapshot_path: source_cif_hash,
    }
    assert_frozen_sources_unchanged()
    assert_paths_unchanged(expected_input_paths)
    cfg['cif_path'] = str(snapshot_path)
    opt_in = resolve_optimization_input(cfg)
    assert_paths_unchanged(expected_input_paths)
    input_hash = manifest_sha256(opt_in.manifest)
    resolved_structure_hash = structure_sha256(opt_in.reference)
    relax = deepcopy(cfg.get('relax_params', {}))
    calc_manifest = calculator_manifest(relax)
    relax['calc_name'] = MODEL_NAME
    expected_provenance = {
        'backend': f'hermes-local-online-strict-isolation-all15-{surrogate}',
        'surrogate_family': ({
            'extra_trees': 'ExtraTrees ensemble on CE-inspired cluster correlations',
            'bayesian_ridge': 'Bayesian ridge on CE-inspired cluster correlations',
            'ard': 'sparse ARD on CE-inspired cluster correlations',
            'aliegnn_online': 'compact ALIEGNN-inspired atom/line-graph energy ensemble trained from scratch',
        }[surrogate]),
        'surrogate_kind': surrogate,
        'seed': seed, 'configured_step_limit': MAX_PROPOSALS,
        **source_hashes,
        'config_sha256': config_hash,
        'source_cif_path': str(source_path),
        'source_cif_sha256': source_cif_hash,
        'frozen_input_snapshot_path': str(snapshot_path),
        'frozen_input_snapshot_sha256': source_cif_hash,
        'input_manifest_sha256': input_hash,
        'resolved_structure_sha256': resolved_structure_hash,
        'calculator_manifest': calc_manifest,
    }
    if summary_path.exists():
        raise RuntimeError(
            f'{system}: completed output exists and automatic reuse is disabled; '
            'verify it externally or choose a clean output directory')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'{system}: nonempty incomplete output must be archived before a clean run: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    run_started_ns = time.monotonic_ns()

    print_loading_info(opt_in, cfg)
    species, ratio = opt_in.atom_swaps, opt_in.ratio
    if len(species) != 2:
        raise ValueError(f'{system}: elite-v2 currently supports binary occupancy only')
    vacancy = vacancy_symbol_from_swaps(species) or 'X'
    template_sites, framework = split_lattice_and_framework(opt_in.reference, opt_in.lattice_element)
    counts = resolve_counts(len(template_sites), species, ratio)
    framework_counts = occupancy_counts(framework.get_chemical_symbols())
    expected_provenance['target_invariants'] = {
        'active_sites': len(template_sites),
        'species': list(species),
        'counts': counts,
        'framework_counts': framework_counts,
    }
    prototype = [species[0]] * counts[species[0]] + [species[1]] * counts[species[1]]
    jump_spaces = {size: jump_space_size(prototype, size) for size in JUMP_SIZES}
    oversized = {size: total for size, total in jump_spaces.items()
                 if total > MAX_EXHAUSTIVE_JUMP_SPACE}
    if oversized:
        raise ValueError(f'{system}: configured jump spaces exceed supported exhaustive limit: {oversized}')
    minority = min(counts, key=counts.get)
    (distances, shells, pair_shell, triangle_index,
     pair_multiplicity, triangle_multiplicity) = geometry(template_sites)
    feature_fn = lambda occ: cluster_correlation_descriptor(
        occ, minority, len(shells), pair_shell, triangle_index,
        pair_multiplicity, triangle_multiplicity)
    aliegnn_geometry = FixedLineGraphGeometry.from_atoms(template_sites, cutoff=8.0, max_neighbors=12)
    graph_digest = hashlib.sha256()
    for array in (aliegnn_geometry.edge_sources, aliegnn_geometry.edge_targets,
                  aliegnn_geometry.distances, aliegnn_geometry.line_sources,
                  aliegnn_geometry.line_targets, aliegnn_geometry.angle_cosines):
        value = np.ascontiguousarray(array)
        graph_digest.update(str(value.dtype).encode())
        graph_digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
        graph_digest.update(value.tobytes())
    expected_provenance['aliegnn_online_contract'] = {
        'architecture': 'compact atom/line-graph invariant energy ensemble',
        'pretrained_checkpoint': None,
        'ensemble_size': 3, 'hidden_dim': 16, 'layers': 1, 'epochs_per_fit': 20,
        'cutoff_angstrom': 8.0, 'max_neighbors': 12,
        'num_nodes': aliegnn_geometry.num_nodes,
        'num_edges': aliegnn_geometry.num_edges,
        'num_line_edges': aliegnn_geometry.num_line_edges,
        'graph_sha256': graph_digest.hexdigest(),
    }

    assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
    (out/'Skill.md').write_bytes(FROZEN_SOURCE_BYTES['skill_sha256'])
    (out/'agentic_policy_v2.py').write_bytes(FROZEN_SOURCE_BYTES['base_policy_sha256'])
    (out/'agentic_policy_ce.py').write_bytes(FROZEN_SOURCE_BYTES['ce_policy_sha256'])
    (out/'agentic_policy_aliegnn_online.py').write_bytes(FROZEN_SOURCE_BYTES['aliegnn_policy_sha256'])
    (out/'run_all15_four_surrogates_five_seeds_frozen_v1.py').write_bytes(
        FROZEN_SOURCE_BYTES['runner_sha256'])
    (out/'utils.py').write_bytes(FROZEN_SOURCE_BYTES['utils_sha256'])
    assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
    write_input_manifest(out, opt_in.manifest)
    writer = SwapStructureWriter(out, species)
    relax['traj_file'] = str(out/'relaxation.xyz')
    optimizer = setup_optimizer(**relax)
    expected_provenance['calculator_state_sha256'] = calculator_state_sha256(optimizer)
    expected_model_hash = expected_provenance['calculator_state_sha256']
    rng = np.random.default_rng(seed)

    initial = initial_occupancy(len(template_sites), species, ratio, rng)
    if occupancy_counts(initial) != counts:
        raise RuntimeError(f'{system}: initial occupancy counts {occupancy_counts(initial)} != required {counts}')
    assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
    assert_calculator_state_unchanged(optimizer, expected_model_hash)
    initial_eval_started_ns = time.monotonic_ns()
    initial_energy, _, initial_structure = evaluate_occupancy(
        initial, template_sites, framework, optimizer, vacancy_symbol=vacancy)
    initial_evaluation_seconds = (time.monotonic_ns() - initial_eval_started_ns) / 1e9
    assert_calculator_state_unchanged(optimizer, expected_model_hash)
    assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
    initial_energy = float(initial_energy)
    write(out/'initial.xyz', initial_structure)
    writer.write(initial_structure, label='step_000_initial', occupancy=initial, energy=initial_energy)
    best_occ, best_energy, best_structure = initial.copy(), initial_energy, initial_structure
    write_atoms_xyz(out/'best.xyz', best_structure)

    seen = {tuple(initial)}
    history_occ = [initial.copy()]
    history_energy = [initial_energy]
    best_elapsed_seconds = (time.monotonic_ns() - run_started_ns) / 1e9
    records = [{'step': 0, 'energy': initial_energy, 'best_energy_eV': initial_energy,
                'occupancy': initial, 'swaps': [], 'num_swaps': 0, 'accepted': True,
                'selection_seconds': 0.0, 'evaluation_seconds': initial_evaluation_seconds,
                'elapsed_seconds': best_elapsed_seconds,
                'rationale': 'initial'}]
    jump_mode_counts = defaultdict(int)
    exhausted_jump_slots = set()
    local_counts = defaultdict(int)
    last_jump_step = 0
    last_improvement_step = 0
    elite_signature = ()
    stable_elite_steps = 0
    unscored_model_jumps = 0
    step = 0
    print(f'[{system}] initial={initial_energy:.8f}; sites={len(template_sites)} counts={counts}', flush=True)

    while True:
        elites = diverse_elite_archive(history_occ, history_energy, MAX_ELITES, MIN_ELITE_HAMMING)
        signature = tuple(elite.fingerprint for elite in elites)
        if signature == elite_signature:
            stable_elite_steps += 1
        else:
            elite_signature = signature
            stable_elite_steps = 0

        unseen_by_elite = {
            elite.fingerprint: single_neighbors_from_parent(elite.occupancy, seen, distances)
            for elite in elites
        }
        unseen_counts = {fp: len(candidates) for fp, candidates in unseen_by_elite.items()}
        evidence = elite_stop_evidence(
            step, step-last_improvement_step, list(signature), unseen_counts,
            jump_mode_counts, exhausted_jump_slots, JUMP_SIZES, ACQUISITIONS,
            stable_elite_steps,
        )
        evidence['all_model_ranked_jumps_have_predictions'] = unscored_model_jumps == 0
        evidence['unscored_model_jumps'] = unscored_model_jumps
        evidence['elite_fingerprints'] = [list(fp) for fp in signature]
        evidence['unseen_single_counts'] = [
            {'parent_fingerprint': list(fp), 'count': unseen_counts[fp]} for fp in signature
        ]
        evidence['done'] = bool(evidence['done'] and evidence['all_model_ranked_jumps_have_predictions'])
        if step >= MAX_PROPOSALS:
            stop_decision = {
                'done': True,
                'converged_by_evidence': bool(evidence['done']),
                'termination_kind': 'hard_proposal_limit',
                'step': step,
                'reason': f'User-specified hard upper limit of {MAX_PROPOSALS} proposals reached.',
                'evidence': evidence,
            }
            break
        if evidence['done']:
            stop_decision = {
                'done': True,
                'converged_by_evidence': True,
                'termination_kind': 'evidence_convergence',
                'step': step,
                'reason': 'Skill.md elite-v2 convergence: the stable diverse elite archive has complete single-swap neighborhoods and every required acquisition mode at balanced jump sizes 2, 3, and 4 was evaluated or its finite slot was proven exhausted, with no global improvement for 40 proposals.',
                'evidence': evidence,
            }
            break

        all_local = [(elite, candidate) for elite in elites
                     for candidate in unseen_by_elite[elite.fingerprint]]
        unmet_slots = sorted(
            [(sum(jump_mode_counts[(elite.fingerprint, size, mode)] for mode in ACQUISITIONS),
              elite.energy, size, ACQUISITIONS.index(mode), elite, mode)
             for elite in elites for size in JUMP_SIZES
             if (elite.fingerprint, size) not in exhausted_jump_slots
             for mode in ACQUISITIONS
             if jump_mode_counts[(elite.fingerprint, size, mode)] < 1],
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )
        jump_due = step >= 8 and bool(unmet_slots) and (
            step-last_jump_step >= JUMP_INTERVAL or not all_local)

        candidates = []
        parent = None
        mode = ''
        acquisition = 'lcb'
        if jump_due:
            for _, _, size, _, candidate_parent, required_mode in unmet_slots:
                proposed = balanced_jump_neighbors(candidate_parent.occupancy, seen, size, rng, limit=3000)
                if proposed:
                    candidates = proposed
                    parent = candidate_parent
                    acquisition = required_mode
                    mode = f'elite-{size}-swap-{acquisition}'
                    last_jump_step = step + 1
                    break
                if jump_space_exhausted(candidate_parent.occupancy, seen, size):
                    exhausted_jump_slots.add((candidate_parent.fingerprint, size))

        if not candidates and all_local:
            parent = min(
                (elite for elite in elites if unseen_by_elite[elite.fingerprint]),
                key=lambda elite: (local_counts[elite.fingerprint], elite.energy),
            )
            candidates = unseen_by_elite[parent.fingerprint]
            if local_counts[parent.fingerprint] < 8:
                minimum = min(candidate.pair_distance for candidate in candidates)
                candidates = [candidate for candidate in candidates
                              if candidate.pair_distance <= minimum + 1e-3]
            acquisition = 'uncertainty' if step and step % 17 == 0 else ('greedy' if step and step % 29 == 0 else 'lcb')
            mode = f'elite-single-{acquisition}'

        if not candidates:
            # Complete plateau evidence with any unseen configured jump, visiting
            # the full elite × size product rather than diagonal pairings.
            for candidate_parent in elites:
                for size in JUMP_SIZES:
                    proposed = balanced_jump_neighbors(candidate_parent.occupancy, seen, size, rng, limit=3000)
                    if not proposed:
                        if jump_space_exhausted(candidate_parent.occupancy, seen, size):
                            exhausted_jump_slots.add((candidate_parent.fingerprint, size))
                        continue
                    candidates = proposed
                    parent = candidate_parent
                    missing_modes = [name for name in ACQUISITIONS
                                     if jump_mode_counts[(parent.fingerprint, size, name)] < 1]
                    acquisition = missing_modes[0] if missing_modes else ACQUISITIONS[step % len(ACQUISITIONS)]
                    mode = f'elite-{size}-swap-consolidation-{acquisition}'
                    last_jump_step = step + 1
                    break
                if candidates:
                    break
        if not candidates or parent is None:
            exhausted_evidence = elite_stop_evidence(
                step, step-last_improvement_step, list(signature), unseen_counts,
                jump_mode_counts, exhausted_jump_slots, JUMP_SIZES, ACQUISITIONS,
                stable_elite_steps,
            )
            configured_space_exhausted = (not all_local and all(
                (elite.fingerprint, size) in exhausted_jump_slots
                for elite in elites for size in JUMP_SIZES))
            if configured_space_exhausted and exhausted_evidence['all_elite_single_neighborhoods_evaluated'] and exhausted_evidence['all_elite_jump_modes_met_or_exhausted'] and unscored_model_jumps == 0:
                exhausted_evidence['configured_policy_space_exhausted'] = True
                exhausted_evidence['done'] = True
                stop_decision = {'done': True, 'converged_by_evidence': True,
                    'termination_kind': 'finite_policy_space_exhaustion', 'step': step,
                    'reason': 'Skill.md elite-v2 finite exhaustion: every configured single and balanced 2/3/4-swap candidate around the current elite archive was evaluated or proven unavailable.',
                    'evidence': exhausted_evidence}
                break
            raise RuntimeError(f'{system}: no candidate and exhaustion not proven: {exhausted_evidence}')

        step += 1
        history_size_before = len(history_energy)
        requested_acquisition = acquisition
        acquisition = effective_acquisition(requested_acquisition, history_size_before)
        if acquisition != requested_acquisition:
            mode = f'{mode}-executed-{acquisition}-warmup'
        selection_started_ns = time.monotonic_ns()
        if surrogate == 'extra_trees':
            selection = select_candidate_v2(
                candidates, history_occ, history_energy, feature_fn,
                seed=seed, acquisition_mode=acquisition)
        elif surrogate in ('bayesian_ridge', 'ard'):
            selection = select_candidate_ce(
                candidates, history_occ, history_energy, feature_fn,
                kind=surrogate, acquisition=acquisition, seed=seed)
        else:
            selection = select_candidate_aliegnn(
                candidates, history_occ, history_energy, aliegnn_geometry, species,
                feature_fn, acquisition=acquisition, seed=seed)
        selection_seconds = (time.monotonic_ns() - selection_started_ns) / 1e9
        candidate = candidates[selection.index]
        if occupancy_counts(candidate.occupancy) != counts:
            raise RuntimeError(
                f'{system}: proposal composition {occupancy_counts(candidate.occupancy)} != required {counts}')
        assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
        assert_calculator_state_unchanged(optimizer, expected_model_hash)
        evaluation_started_ns = time.monotonic_ns()
        trial_energy, _, trial_structure = evaluate_occupancy(
            candidate.occupancy, template_sites, framework, optimizer, vacancy_symbol=vacancy)
        evaluation_seconds = (time.monotonic_ns() - evaluation_started_ns) / 1e9
        assert_calculator_state_unchanged(optimizer, expected_model_hash)
        assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
        trial_energy = float(trial_energy)
        seen.add(tuple(candidate.occupancy))
        history_occ.append(candidate.occupancy.copy())
        history_energy.append(trial_energy)
        previous_best = best_energy
        accepted = trial_energy < best_energy
        if candidate.num_swaps == 1:
            local_counts[parent.fingerprint] += 1
        else:
            jump_mode_counts[(parent.fingerprint, candidate.num_swaps, acquisition)] += 1
            if history_size_before >= 8 and acquisition != 'novelty' and selection.predicted_energy is None:
                unscored_model_jumps += 1
        writer.write(trial_structure, label=f'step_{step:04d}',
                     occupancy=candidate.occupancy, energy=trial_energy)
        if accepted:
            best_occ, best_energy, best_structure = candidate.occupancy.copy(), trial_energy, trial_structure
            last_improvement_step = step
            best_elapsed_seconds = (time.monotonic_ns() - run_started_ns) / 1e9
            write_atoms_xyz(out/'best.xyz', best_structure)
            writer.write(best_structure, label='best', occupancy=best_occ, energy=best_energy)

        record = {
            'step': step, 'energy': trial_energy, 'best_energy_eV': best_energy,
            'occupancy': candidate.occupancy, 'swaps': candidate.swaps,
            'num_swaps': candidate.num_swaps, 'accepted': accepted,
            'delta_to_previous_best_eV': trial_energy-previous_best,
            'parent_fingerprint': list(parent.fingerprint),
            'parent_energy_eV': parent.energy,
            'requested_acquisition_mode': requested_acquisition,
            'acquisition_mode': acquisition,
            'predicted_energy_eV': selection.predicted_energy,
            'uncertainty_eV': selection.uncertainty,
            'prequential_residual_eV': (None if selection.predicted_energy is None else
                                        trial_energy - selection.predicted_energy),
            'novelty': selection.novelty, 'selector': selection.selector,
            'surrogate_training_data_sha256': getattr(selection, 'training_data_sha256', None),
            'surrogate_ensemble_state_sha256': getattr(selection, 'ensemble_state_sha256', None),
            'selection_seconds': selection_seconds,
            'evaluation_seconds': evaluation_seconds,
            'elapsed_seconds': (time.monotonic_ns() - run_started_ns) / 1e9,
            'rationale': mode,
        }
        records.append(record)
        partial = {
            'system': system, **expected_provenance,
            'allowed_inputs': ['frozen Skill/policy', 'current config/CIF', 'current-run observations'],
            'best_energy_eV': best_energy, 'best_occupancy': best_occ,
            'best_counts': occupancy_counts(best_occ),
            'elapsed_seconds': (time.monotonic_ns() - run_started_ns) / 1e9,
            'wallclock_to_best_seconds': best_elapsed_seconds,
            'best_first_step': last_improvement_step,
            'elite_archive': [{'occupancy': elite.occupancy, 'energy_eV': elite.energy,
                               'history_index': elite.history_index} for elite in elites],
            'jump_mode_counts': serialize_jump_mode_counts(jump_mode_counts),
            'exhausted_jump_slots': [{'parent_fingerprint': list(fp), 'size': size}
                                     for fp, size in sorted(exhausted_jump_slots)],
            'stop_decision': {'done': False, 'latest_evidence': evidence},
            'steps': records,
        }
        assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
        (out/'summary_partial.json').write_text(json.dumps(partial, indent=2))
        print(f'[{system} step {step}] {mode} E={trial_energy:.8f} best={best_energy:.8f} accepted={accepted}', flush=True)

    final_elites = diverse_elite_archive(history_occ, history_energy, MAX_ELITES, MIN_ELITE_HAMMING)
    summary = {
        'system': system, **expected_provenance,
        'allowed_inputs': ['frozen Skill/policy', 'current config/CIF', 'current-run observations'],
        'best_energy_eV': best_energy, 'best_occupancy': best_occ,
        'best_counts': occupancy_counts(best_occ),
        'elapsed_seconds': (time.monotonic_ns() - run_started_ns) / 1e9,
        'wallclock_to_best_seconds': best_elapsed_seconds,
        'best_first_step': last_improvement_step,
        'elite_archive': [{'occupancy': elite.occupancy, 'energy_eV': elite.energy,
                           'history_index': elite.history_index} for elite in final_elites],
        'jump_mode_counts': serialize_jump_mode_counts(jump_mode_counts),
        'exhausted_jump_slots': [{'parent_fingerprint': list(fp), 'size': size}
                                 for fp, size in sorted(exhausted_jump_slots)],
        'stop_decision': stop_decision, 'steps': records,
    }
    assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
    assert_calculator_state_unchanged(optimizer, expected_model_hash)
    summary_tmp = out/'summary.json.tmp'
    summary_tmp.write_text(json.dumps(summary, indent=2))
    write_atoms_xyz(out/'best.xyz', best_structure)
    writer.write(best_structure, label='best_final', occupancy=best_occ, energy=best_energy)
    assert_frozen_sources_unchanged(); assert_paths_unchanged(expected_input_paths)
    assert_calculator_state_unchanged(optimizer, expected_model_hash)
    summary_tmp.replace(out/'summary.json')
    print(f'[{system} {surrogate} seed={seed}] STOP at {step}; best={best_energy:.8f}', flush=True)
    return summary


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_LOCK.mkdir(exist_ok=False)
    records = []
    try:
        for system_index, system in enumerate(SYSTEMS):
            for seed in SEEDS:
                for surrogate in arm_order(system_index, seed):
                    status = {'system': system, 'surrogate': surrogate, 'seed': seed,
                              'arm_order': list(arm_order(system_index, seed)), 'status': 'active',
                              'started_at': time.time()}
                    records.append(status)
                    write_batch(records)
                    try:
                        summary = run_system(system, surrogate, seed)
                        status.update(status='complete', best_energy_eV=summary['best_energy_eV'],
                                      steps=len(summary['steps'])-1,
                                      stop_decision=summary['stop_decision'], finished_at=time.time())
                    except Exception as exc:
                        status.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                                      finished_at=time.time())
                        write_batch(records)
                        print(f'[{system} {surrogate} seed={seed}] FAILED {exc}', flush=True)
                        raise
                    write_batch(records)
        failures = [row for row in records if row['status'] != 'complete']
        if len(records) != 300 or failures:
            raise RuntimeError(
                f'benchmark campaign finished incomplete: arms={len(records)} failures={failures}')
        print('ALL15 FOUR-SURROGATE FIVE-SEED CAMPAIGN COMPLETE', flush=True)
    except Exception:
        # Retain the lock after any failure so restart requires deliberate inspection/archive.
        raise
    else:
        CAMPAIGN_LOCK.rmdir()


if __name__ == '__main__':
    main()
