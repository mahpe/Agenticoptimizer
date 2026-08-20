"""
Shared utilities for Agenticoptimizer.

Structure loading/preparation, MatCalc relaxation, occupancy helpers, and
disordered CIF preparation used by agentic_optimizer.py, genetic_algo_optimizer.py,
Metropolis_MC.py, and prepare_disordered.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import ase
import matcalc as mtc
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from packaging import version

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser


# FrechetCellFilter moved from ase.constraints to ase.filters in ASE 3.23+.
if version.parse(ase.__version__) > version.parse("3.22.1"):
    from ase.filters import FrechetCellFilter as CellFilter
else:
    from ase.constraints import FrechetCellFilter as CellFilter

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_CIF = ROOT / "NaFePO4_olivine_new.cif"

CELL_FILTERS = {
    "CellFilter": CellFilter,
    "FrechetCellFilter": CellFilter,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Load config.toml; normalize supercell_size lists to tuples for downstream code."""
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if "supercell_size" in cfg and isinstance(cfg["supercell_size"], list):
        cfg["supercell_size"] = tuple(cfg["supercell_size"])
    return cfg


# ---------------------------------------------------------------------------
# Relaxation / structure assembly
# ---------------------------------------------------------------------------


def setup_optimizer(
    calc_name="TensorNet-MatPES-PBE-2025.2",
    optimizer="BFGS",
    interval=1,
    fmax=0.2,
    max_steps=500,
    relax_atoms=True,
    relax_cell=True,
    cell_filter="FrechetCellFilter",
    fix_symmetry=False,
    fix_atoms=False,
    symprec=1e-3,
    traj_file="relaxation.traj",
):
    """Setup the MatCalc relaxer for structure optimization."""
    if isinstance(cell_filter, str):
        cell_filter = CELL_FILTERS.get(cell_filter, CellFilter)

    calculator = mtc.load_fp(calc_name)
    return mtc.RelaxCalc(
        calculator,
        optimizer=optimizer,
        traj_file=traj_file,
        interval=interval,
        fmax=fmax,
        max_steps=max_steps,
        relax_atoms=relax_atoms,
        relax_cell=relax_cell,
        cell_filter=cell_filter,
        fix_symmetry=fix_symmetry,
        fix_atoms=fix_atoms,
        symprec=symprec,
    )


def get_reference_structures(cif_path, supercell_size=(1, 1, 1)):
    """Get the reference supercell from a CIF file."""
    return read(cif_path) * tuple(supercell_size)


def split_lattice_and_framework(structure, lattice_element="Na"):
    """
    Split a structure into lattice-site atoms and the remaining framework.

    Lattice sites are the positions that can be occupied by swap species
    (e.g. Na / vacancy X). The framework is everything else (e.g. FePO4).
    """
    lattice = Atoms(
        [atom for atom in structure if atom.symbol == lattice_element],
        cell=structure.cell,
        pbc=structure.pbc,
    )
    framework = structure.copy()
    lattice_indices = [atom.index for atom in framework if atom.symbol == lattice_element]
    del framework[lattice_indices]
    return lattice, framework


def assemble_structure(site_atoms, framework, vacancy_symbol="X"):
    """
    Combine occupied lattice sites with the framework and drop vacancies.

    ``X`` (or ``vacancy_symbol``) marks empty lattice sites and is removed
    before relaxation, matching the cPaiNN GA convention.
    """
    structure = site_atoms.copy()
    structure += framework
    vacancy_indices = [atom.index for atom in structure if atom.symbol == vacancy_symbol]
    if vacancy_indices:
        del structure[vacancy_indices]
    return structure


def apply_occupancy(template_sites, occupancy):
    """Return a copy of lattice sites with symbols set from ``occupancy``."""
    if len(occupancy) != len(template_sites):
        raise ValueError(
            f"Occupancy length {len(occupancy)} != number of sites {len(template_sites)}"
        )
    sites = template_sites.copy()
    for i, symbol in enumerate(occupancy):
        sites[i].symbol = symbol
    return sites


def relaxed_atoms_from_results(results: dict) -> Atoms:
    """Build relaxed ASE Atoms with SinglePointCalculator from MatCalc relax output."""
    final = results["final_structure"]
    if isinstance(final, Atoms):
        atoms = final.copy()
    else:
        atoms = AseAtomsAdaptor.get_atoms(final)

    calc_kwargs: dict = {"energy": float(results["energy"])}
    if results.get("forces") is not None:
        calc_kwargs["forces"] = np.asarray(results["forces"])
    if results.get("stress") is not None:
        calc_kwargs["stress"] = np.asarray(results["stress"])
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms


def copy_atoms_with_calc(atoms: Atoms) -> Atoms:
    """Copy Atoms and reattach calculator results for trajectory I/O."""
    snapshot = atoms.copy()
    if atoms.calc is not None and atoms.calc.results:
        snapshot.calc = SinglePointCalculator(snapshot, **atoms.calc.results)
    return snapshot


def write_atoms_xyz(path: Path | str, atoms: Atoms) -> None:
    """Write ASE Atoms to .xyz with SinglePointCalculator attached."""
    write(path, copy_atoms_with_calc(atoms))


def get_score(structure, optimizer):
    """Relax a structure and return total energy (eV), results dict, and relaxed Atoms."""
    results = optimizer.calc(structure)
    return results["energy"], results, relaxed_atoms_from_results(results)


def evaluate_occupancy(
    occupancy: list[str],
    template_sites,
    framework,
    optimizer,
    vacancy_symbol: str = "X",
):
    """Apply occupancy, assemble full structure, relax; used by agent and MC."""
    sites = apply_occupancy(template_sites, occupancy)
    structure = assemble_structure(sites, framework, vacancy_symbol=vacancy_symbol)
    energy, results, relaxed = get_score(structure, optimizer)
    return energy, results, relaxed


class SwapStructureWriter:
    """Append relaxed configurations to one .xyz trajectory (SinglePointCalculator preserved)."""

    def __init__(self, out_dir: Path | str, atom_swaps: list[str]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.atom_swaps = atom_swaps
        self.trajectory = self.out_dir / "atom_swaps.xyz"
        self._frame = 0
        if self.trajectory.exists():
            self.trajectory.unlink()

    def write(
        self,
        atoms: Atoms,
        *,
        label: str,
        occupancy: list[str] | None = None,
        energy: float | None = None,
    ) -> Path:
        snapshot = copy_atoms_with_calc(atoms)
        snapshot.info["atom_swaps"] = ",".join(self.atom_swaps)
        snapshot.info["label"] = label
        if occupancy is not None:
            snapshot.info["occupancy"] = "".join(occupancy)
        if energy is None and snapshot.calc is not None:
            try:
                energy = float(snapshot.get_potential_energy())
            except RuntimeError:
                energy = None
        if energy is not None:
            snapshot.info["energy_eV"] = energy

        write(self.trajectory, snapshot, append=True)
        self._frame += 1
        return self.trajectory


def relax_lattice_sites(
    lattice_atoms: Atoms,
    framework: Atoms,
    optimizer,
    vacancy_symbol: str = "X",
) -> tuple[float, Atoms]:
    """Assemble lattice + framework, relax, return total energy (eV) and relaxed Atoms."""
    structure = assemble_structure(lattice_atoms, framework, vacancy_symbol=vacancy_symbol)
    energy, _results, relaxed = get_score(structure, optimizer)
    return energy, relaxed


def lattice_only(structure: Atoms, keep_symbol: str) -> Atoms:
    """Return a copy with only ``keep_symbol`` sites (cPaiNN GA convention)."""
    indices = [atom.index for atom in structure if atom.symbol != keep_symbol]
    lattice = structure.copy()
    if indices:
        del lattice[indices]
    return lattice


def build_endpoint_lattices(
    reference: Atoms,
    lattice_element: str,
    vacancy_symbol: str | None = None,
    alternate_symbol: str | None = None,
) -> tuple[Atoms, Atoms, int]:
    """
    Build lattice-only endpoint structures for mixing-energy references.

    - Vacancy systems (Na/X): ``lattice_element`` fully occupied vs all ``X``
    - Binary alloys (Au/Cu): ``lattice_element`` fully occupied vs all ``alternate_symbol``
    """
    endpoint_a = lattice_only(reference, lattice_element)
    endpoint_b = endpoint_a.copy()
    if vacancy_symbol is not None:
        for atom in endpoint_b:
            if atom.symbol == lattice_element:
                atom.symbol = vacancy_symbol
    elif alternate_symbol is not None:
        for atom in endpoint_b:
            if atom.symbol == lattice_element:
                atom.symbol = alternate_symbol
    else:
        raise ValueError(
            "build_endpoint_lattices requires vacancy_symbol or alternate_symbol"
        )
    fu = len([atom for atom in endpoint_a if atom.symbol == lattice_element])
    return endpoint_a, endpoint_b, fu


# ---------------------------------------------------------------------------
# Occupancy helpers
# ---------------------------------------------------------------------------


def resolve_counts(n_sites: int, species: list[str], ratio: list[float]) -> dict[str, int]:
    """Largest-remainder rounding: integer counts that sum to n_sites and match ratio."""
    if len(species) != len(ratio):
        raise ValueError("atom_swaps and ratio must have the same length")
    if abs(sum(ratio) - 1.0) > 1e-6:
        raise ValueError(f"ratio must sum to 1, got {sum(ratio)}")

    raw = [r * n_sites for r in ratio]
    counts = [int(np.floor(x)) for x in raw]
    remainders = sorted(
        ((raw[i] - counts[i], i) for i in range(len(species))),
        reverse=True,
    )
    leftover = n_sites - sum(counts)
    for k in range(leftover):
        counts[remainders[k][1]] += 1
    return {species[i]: counts[i] for i in range(len(species))}


def initial_occupancy(
    n_sites: int,
    species: list[str],
    ratio: list[float],
    rng: np.random.Generator,
) -> list[str]:
    """Random permutation at fixed composition (counts from resolve_counts)."""
    counts = resolve_counts(n_sites, species, ratio)
    occ = []
    for sym, n in counts.items():
        occ.extend([sym] * n)
    rng.shuffle(occ)
    return occ


def apply_swaps(occupancy: list[str], swaps: list[list[int]]) -> list[str]:
    """Apply one or more [i, j] exchanges; used by the agent backend."""
    occ = occupancy.copy()
    n = len(occ)
    for pair in swaps:
        if len(pair) != 2:
            raise ValueError(f"Swap must be [i, j], got {pair}")
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Swap indices out of range: {pair} for N={n}")
        occ[i], occ[j] = occ[j], occ[i]
    return occ


def apply_swap(occupancy: list[str], i: int, j: int) -> list[str]:
    occ = occupancy.copy()
    occ[i], occ[j] = occ[j], occ[i]
    return occ


def occupancy_counts(occupancy: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym in occupancy:
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def vacancy_symbol_from_swaps(atom_swaps: list[str]) -> str | None:
    """Return ``X`` when vacancies are part of the swap list, else ``None``."""
    return "X" if "X" in atom_swaps else None


def unlike_site_pairs(occupancy: list[str]) -> list[tuple[int, int]]:
    """All index pairs (i, j) where occupancy differs — valid MC swap candidates."""
    return [
        (i, j)
        for i in range(len(occupancy))
        for j in range(i + 1, len(occupancy))
        if occupancy[i] != occupancy[j]
    ]


def propose_swap_from_pairs(pairs: list[tuple[int, int]], rng: np.random.Generator) -> tuple[int, int]:
    """Pick one unlike pair uniformly at random."""
    idx = int(rng.integers(len(pairs)))
    return pairs[idx]


def fixed_rate_accept(
    delta_e: float, acceptance_rate: float, rng: np.random.Generator
) -> tuple[bool, float]:
    """Accept improving moves; accept uphill moves with fixed probability."""
    if delta_e <= 0.0:
        return True, 1.0
    if acceptance_rate <= 0.0:
        return False, 0.0
    if acceptance_rate >= 1.0:
        return True, 1.0
    return bool(rng.random() < acceptance_rate), acceptance_rate


def metropolis_accept(delta_e: float, temperature: float, rng: np.random.Generator) -> tuple[bool, float]:
    """Standard Metropolis rule; returns (accepted, acceptance_probability)."""
    if delta_e <= 0.0:
        return True, 1.0
    if temperature <= 0.0:
        return False, 0.0
    prob = math.exp(-delta_e / temperature)
    return bool(rng.random() < prob), prob


# ---------------------------------------------------------------------------
# Disordered-structure preparation
# ---------------------------------------------------------------------------


@dataclass
class OptimizationInput:
    """Resolved structure + swap metadata passed from config to all optimizers."""

    reference: Atoms  # full ASE structure (already supercelled when prepared=True)
    lattice_element: str  # placeholder symbol on swap sites before assignment
    atom_swaps: list[str]
    ratio: list[float]
    supercell_size: tuple[int, int, int]  # (1,1,1) when reference is already expanded
    prepared: bool = False
    min_cell_size: int = 1  # a-axis replica factor used during disordered prep
    n_swap_sites: int = 0
    sublattice_label: str = ""
    manifest: dict = field(default_factory=dict)


def _species_element(species) -> str:
    """Strip oxidation state from pymatgen species (e.g. Fe2+ -> Fe)."""
    if hasattr(species, "symbol"):
        return str(species.symbol)
    text = str(species)
    return text.split("+")[0].split("-")[0]


def site_species_dict(site) -> dict[str, float]:
    """Per-site occupancies; unfilled site fraction becomes vacancy symbol X."""
    counts: dict[str, float] = {}
    for species, amount in site.species.items():
        element = _species_element(species)
        counts[element] = counts.get(element, 0.0) + float(amount)
    vacancy = 1.0 - sum(counts.values())
    if vacancy > 1e-5:
        counts["X"] = counts.get("X", 0.0) + vacancy
    return counts


def site_signature(site) -> frozenset[tuple[str, float]]:
    """Hashable occupancy fingerprint for grouping equivalent disorder sites."""
    return frozenset((el, round(occ, 6)) for el, occ in site_species_dict(site).items())


def parse_cif_structure(cif_path: Path | str, occupancy_tolerance: float = 2.0) -> Structure:
    """Parse first structure from a CIF (conventional cell, not primitive)."""
    parser = CifParser(str(cif_path), occupancy_tolerance=occupancy_tolerance)
    structures = parser.parse_structures(primitive=False)
    if not structures:
        raise ValueError(f"No structure parsed from {cif_path}")
    return structures[0]


def identify_disorder_sublattices(structure: Structure) -> list[dict]:
    """Group disordered sites by identical occupancy signature."""
    sig_to_indices: dict[frozenset, list[int]] = {}
    for index, site in enumerate(structure):
        if site.is_ordered:
            continue
        signature = site_signature(site)
        sig_to_indices.setdefault(signature, []).append(index)

    sublattices = []
    for signature, indices in sig_to_indices.items():
        label = "".join(
            f"{element}{occ:.4g}".rstrip("0").rstrip(".")
            for element, occ in sorted(signature, key=lambda item: item[0])
        )
        sublattices.append(
            {
                "signature": signature,
                "site_indices": sorted(indices),
                "label": label or "disorder",
            }
        )
    sublattices.sort(key=lambda item: item["label"])
    return sublattices


def min_supercell_size_for_sites(structure: Structure, site_indices: list[int], max_cell_size: int) -> int:
    """Smallest replica factor n (a-axis) giving integer species totals on chosen sites."""
    totals: dict[str, float] = {}
    for index in site_indices:
        for element, amount in site_species_dict(structure[index]).items():
            totals[element] = totals.get(element, 0.0) + amount

    if not totals:
        return 1

    # Try smallest n so n × (per-cell fractional totals) are all integers.
    for n in range(1, max_cell_size + 1):
        if all(abs(n * total - round(n * total)) < 1e-6 for total in totals.values()):
            return n

    # Fallback: LCM of fractional denominators when brute-force search fails.
    denominators = []
    for index in site_indices:
        amounts = list(site_species_dict(structure[index]).values())
        denominators.extend(
            Fraction(float(amount)).limit_denominator(max_cell_size).denominator
            for amount in amounts
        )
    return math.lcm(*denominators) if denominators else 1


def species_and_ratio_from_signature(signature: frozenset[tuple[str, float]]) -> tuple[list[str], list[float]]:
    """Convert a disorder signature to sorted ``atom_swaps`` and ``ratio`` lists."""
    ordered = sorted(signature, key=lambda item: item[0])
    atom_swaps = [element for element, _occ in ordered]
    # Keep vacancy X last so real elements map to lattice_element first.
    if "X" in atom_swaps:
        atom_swaps = [element for element in atom_swaps if element != "X"] + ["X"]
    occ_map = dict(signature)
    ratio = [occ_map[element] for element in atom_swaps]
    return atom_swaps, ratio


def normalize_supercell_size(value) -> tuple[int, int, int]:
    """Accept int, list, or tuple and return ``(nx, ny, nz)``."""
    if value is None:
        raise ValueError("supercell_size is required")
    if isinstance(value, int):
        return (value, 1, 1)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), 1, 1)
        if len(value) == 3:
            return tuple(int(x) for x in value)
        raise ValueError(f"supercell_size must have length 1 or 3, got {value}")
    raise TypeError(f"Invalid supercell_size type: {type(value)}")


def supercell_scaling_matrix(supercell_size: tuple[int, int, int]) -> list[list[int]]:
    nx, ny, nz = supercell_size
    return [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]


def min_supercell_for_ratio(
    n_sites_per_cell: int,
    species: list[str],
    ratio: list[float],
    max_cell_size: int,
) -> int:
    """
    Smallest a-axis replica factor ``n`` so ``n * n_sites_per_cell`` swap sites
    can realize ``ratio`` with integer atom counts.
    """
    if abs(sum(ratio) - 1.0) > 1e-6:
        raise ValueError(f"ratio must sum to 1, got {sum(ratio)}")

    for n in range(1, max_cell_size + 1):
        total_sites = n_sites_per_cell * n
        if all(abs(ratio[i] * total_sites - round(ratio[i] * total_sites)) < 1e-6 for i in range(len(species))):
            counts = resolve_counts(total_sites, species, ratio)
            actual = [counts[species[i]] / total_sites for i in range(len(species))]
            if all(abs(actual[i] - ratio[i]) < 1e-6 for i in range(len(species))):
                return n

    raise ValueError(
        f"No supercell factor 1..{max_cell_size} achieves ratio {ratio} "
        f"with {n_sites_per_cell} swap sites per cell"
    )


def validate_supercell_for_ratio(
    n_sites_per_cell: int,
    supercell_size: tuple[int, int, int],
    species: list[str],
    ratio: list[float],
) -> int:
    """Return total swap sites and raise if ``ratio`` is not integer-realizable."""
    nx, ny, nz = supercell_size
    total_sites = n_sites_per_cell * nx * ny * nz
    counts = resolve_counts(total_sites, species, ratio)
    actual = [counts[species[i]] / total_sites for i in range(len(species))]
    if not all(abs(actual[i] - ratio[i]) < 1e-6 for i in range(len(species))):
        raise ValueError(
            f"supercell_size={supercell_size} gives {total_sites} swap sites, "
            f"which cannot realize ratio {ratio} (actual {actual})"
        )
    return total_sites


def dominant_species(site) -> str:
    """Highest-occupancy element on a partially occupied site (for freezing non-active sublattices)."""
    return max(site_species_dict(site).items(), key=lambda item: item[1])[0]


def ordered_ase_from_supercell(
    supercell: Structure,
    swap_indices: list[int],
    other_disordered_signatures: set[frozenset[tuple[str, float]]],
    placeholder: str,
) -> Atoms:
    """Build an ordered ASE Atoms object from a disordered pymatgen supercell."""
    symbols: list[str] = []
    frac_coords = []
    for index, site in enumerate(supercell):
        if index in swap_indices:
            symbol = placeholder  # active sublattice: filled later by optimizers
        elif not site.is_ordered and site_signature(site) in other_disordered_signatures:
            symbol = dominant_species(site)  # freeze other disorder groups at majority species
        elif site.is_ordered:
            symbol = _species_element(next(iter(site.species.keys())))
        else:
            symbol = dominant_species(site)
        symbols.append(symbol)
        frac_coords.append(site.frac_coords)

    return Atoms(
        symbols=symbols,
        scaled_positions=frac_coords,
        cell=supercell.lattice.matrix,
        pbc=True,
    )


def prepare_disordered_input(
    cif_path: Path | str,
    max_cell_size: int = 20,
    sublattice_index: int = 0,
    occupancy_tolerance: float = 2.0,
    lattice_element: str | None = None,
    atom_swaps: list[str] | None = None,
    ratio: list[float] | None = None,
    supercell_size: tuple[int, int, int] | None = None,
) -> OptimizationInput | None:
    """
    Expand a disordered CIF to an integer-occupancy supercell ready for optimizers.

    Species/ratio default to the CIF disorder signature. Pass ``atom_swaps``,
    ``ratio``, and/or ``supercell_size`` to override (manual concentration / sizing).

    Returns None when the CIF has no partial occupancies.
    """
    structure = parse_cif_structure(cif_path, occupancy_tolerance=occupancy_tolerance)
    sublattices = identify_disorder_sublattices(structure)
    if not sublattices:
        return None

    if not (0 <= sublattice_index < len(sublattices)):
        raise IndexError(
            f"sublattice_index {sublattice_index} out of range for "
            f"{len(sublattices)} disorder group(s)"
        )

    active = sublattices[sublattice_index]
    active_signature = active["signature"]
    active_indices = active["site_indices"]
    n_sites_per_cell = len(active_indices)

    cif_swaps, cif_ratio = species_and_ratio_from_signature(active_signature)
    manual_ratio_provided = ratio is not None
    manual_swaps_provided = atom_swaps is not None
    atom_swaps = [str(s) for s in atom_swaps] if manual_swaps_provided else cif_swaps
    ratio = [float(x) for x in ratio] if manual_ratio_provided else cif_ratio
    ratio_source = "manual" if manual_ratio_provided else "cif"

    if supercell_size is None:
        if ratio_source == "manual":
            n_a = min_supercell_for_ratio(n_sites_per_cell, atom_swaps, ratio, max_cell_size)
            supercell_size = (n_a, 1, 1)
        else:
            n_a = min_supercell_size_for_sites(structure, active_indices, max_cell_size)
            supercell_size = (n_a, 1, 1)

    validate_supercell_for_ratio(n_sites_per_cell, supercell_size, atom_swaps, ratio)

    scaling = supercell_scaling_matrix(supercell_size)
    supercell = structure.make_supercell(scaling)

    swap_indices: list[int] = []
    for index, site in enumerate(supercell):
        if site_signature(site) == active_signature:
            swap_indices.append(index)

    if not swap_indices:
        raise RuntimeError("No swap sites found in prepared supercell.")

    n_swap = len(swap_indices)
    counts = resolve_counts(n_swap, atom_swaps, ratio)
    totals = {species: float(counts[species]) for species in atom_swaps}

    # Placeholder on swap sites = most abundant real species (cPaiNN GA convention).
    if lattice_element is None:
        real_swaps = [element for element in atom_swaps if element != "X"]
        lattice_element = max(real_swaps, key=lambda element: totals.get(element, 0.0))

    other_disordered = {
        sub["signature"]
        for sub in sublattices
        if sub["signature"] != active_signature
    }
    atoms = ordered_ase_from_supercell(
        supercell,
        swap_indices=swap_indices,
        other_disordered_signatures=other_disordered,
        placeholder=lattice_element,
    )

    manifest = {
        "input_cif": str(Path(cif_path).resolve()),
        "prepared": True,
        "supercell_size": list(supercell_size),
        "n_swap_sites": n_swap,
        "sublattice_index": sublattice_index,
        "sublattice_label": active["label"],
        "all_sublattices": [sub["label"] for sub in sublattices],
        "atom_swaps": atom_swaps,
        "ratio": ratio,
        "ratio_source": ratio_source,
        "lattice_element": lattice_element,
        "species_counts": counts,
        "species_totals": totals,
    }

    return OptimizationInput(
        reference=atoms,
        lattice_element=lattice_element,
        atom_swaps=atom_swaps,
        ratio=ratio,
        supercell_size=(1, 1, 1),  # expansion already baked into reference
        prepared=True,
        min_cell_size=supercell_size[0],
        n_swap_sites=n_swap,
        sublattice_label=active["label"],
        manifest=manifest,
    )


def load_prepared_from_manifest(cif_path: Path) -> OptimizationInput | None:
    """Load a prepared supercell when ``manifest.json`` sits beside ``cif_path``."""
    manifest_path = cif_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("prepared"):
        return None

    reference = read(str(cif_path))
    supercell_size = manifest.get("supercell_size", [1, 1, 1])
    return OptimizationInput(
        reference=reference,
        lattice_element=str(manifest["lattice_element"]),
        atom_swaps=[str(s) for s in manifest["atom_swaps"]],
        ratio=[float(x) for x in manifest["ratio"]],
        supercell_size=(1, 1, 1),
        prepared=True,
        min_cell_size=int(supercell_size[0]) if supercell_size else 1,
        n_swap_sites=int(manifest["n_swap_sites"]),
        sublattice_label=str(manifest.get("sublattice_label", "")),
        manifest=manifest,
    )


def resolve_optimization_input(cfg: dict) -> OptimizationInput:
    """
    Load a structure for the optimizers.

    Disordered CIF (``prep_params.auto_prepare = true``, default):
      - Reads partial occupancies from ``cif_path``
      - Builds the smallest (or user-specified) supercell with integer counts
      - Sets ``atom_swaps`` / ``ratio`` from the CIF

    Prepared run (``auto_prepare = false`` + ``manifest.json`` next to ``cif_path``):
      - Loads ``prepared.cif`` and swap metadata from the manifest

    Manual mode (``auto_prepare = false``, ordered CIF):
      - Uses ``cif_path`` + ``supercell_size`` + optional ``[swap_params]`` overrides
    """
    prep_cfg = cfg.get("prep_params", {})
    auto_prepare = bool(prep_cfg.get("auto_prepare", True))
    cif_path = Path(cfg.get("cif_path", DEFAULT_CIF))
    swap = cfg.get("swap_params", {})

    manual_supercell = prep_cfg.get("supercell_size", cfg.get("supercell_size"))
    manual_supercell = (
        normalize_supercell_size(manual_supercell) if manual_supercell is not None else None
    )
    manual_swaps = [str(s) for s in swap["atom_swaps"]] if swap.get("atom_swaps") else None
    manual_ratio = [float(x) for x in swap["ratio"]] if swap.get("ratio") else None
    manual_lattice = swap.get("lattice_sites")

    if auto_prepare:
        # Default path: read disorder from CIF, build integer-occupancy supercell.
        prepared = prepare_disordered_input(
            cif_path=cif_path,
            max_cell_size=int(prep_cfg.get("max_cell_size", 20)),
            sublattice_index=int(prep_cfg.get("sublattice_index", 0)),
            occupancy_tolerance=float(prep_cfg.get("occupancy_tolerance", 2.0)),
            lattice_element=manual_lattice,
            atom_swaps=manual_swaps,
            ratio=manual_ratio,
            supercell_size=manual_supercell,
        )
        if prepared is not None:
            print(
                f"Prepared disordered input: supercell={prepared.manifest.get('supercell_size')}, "
                f"{prepared.n_swap_sites} swap sites ({prepared.sublattice_label})"
            )
            print(f"Swap species: {prepared.atom_swaps}, ratio: {prepared.ratio} "
                  f"({prepared.manifest.get('ratio_source', 'cif')})")
            return prepared
    else:
        prepared = load_prepared_from_manifest(cif_path)
        if prepared is not None:
            print(
                f"Loaded prepared input from {cif_path.parent / 'manifest.json'}: "
                f"{prepared.n_swap_sites} swap sites ({prepared.sublattice_label})"
            )
            print(f"Swap species: {prepared.atom_swaps}, ratio: {prepared.ratio}")
            return prepared

        # Disordered CIF without prior prepare_disordered.py run: still infer swaps from CIF.
        prepared = prepare_disordered_input(
            cif_path=cif_path,
            max_cell_size=int(prep_cfg.get("max_cell_size", 20)),
            sublattice_index=int(prep_cfg.get("sublattice_index", 0)),
            occupancy_tolerance=float(prep_cfg.get("occupancy_tolerance", 2.0)),
            lattice_element=manual_lattice,
            atom_swaps=manual_swaps,
            ratio=manual_ratio,
            supercell_size=manual_supercell,
        )
        if prepared is not None:
            print(
                f"Prepared disordered input (manual prep): "
                f"supercell={prepared.manifest.get('supercell_size')}, "
                f"{prepared.n_swap_sites} swap sites ({prepared.sublattice_label})"
            )
            print(f"Swap species: {prepared.atom_swaps}, ratio: {prepared.ratio} "
                  f"({prepared.manifest.get('ratio_source', 'cif')})")
            return prepared

    # Ordered CIF: user must supply swap_params or a prepared manifest.
    if manual_supercell is None:
        manual_supercell = (1, 1, 1)
    if not manual_swaps or not manual_ratio or not manual_lattice:
        raise ValueError(
            "Manual mode (auto_prepare=false) requires a prepared manifest.json "
            "beside cif_path, or [swap_params] with lattice_sites, atom_swaps, "
            "and ratio, or use auto_prepare=true with a disordered CIF."
        )

    reference = get_reference_structures(str(cif_path), supercell_size=manual_supercell)
    template_sites, _framework = split_lattice_and_framework(reference, manual_lattice)
    n_sites = len(template_sites)
    if n_sites == 0:
        raise ValueError(
            f"No swap sites with placeholder '{manual_lattice}' found in {cif_path}. "
            "Use a prepared CIF or check lattice_sites."
        )
    counts = resolve_counts(n_sites, manual_swaps, manual_ratio)
    actual = [counts[manual_swaps[i]] / n_sites for i in range(len(manual_swaps))]
    if not all(abs(actual[i] - manual_ratio[i]) < 1e-6 for i in range(len(manual_swaps))):
        raise ValueError(
            f"{n_sites} swap sites in manual supercell cannot realize ratio {manual_ratio} "
            f"(actual {actual}). Increase supercell_size."
        )

    return OptimizationInput(
        reference=reference,
        lattice_element=str(manual_lattice),
        atom_swaps=manual_swaps,
        ratio=manual_ratio,
        supercell_size=manual_supercell,
        prepared=False,
        n_swap_sites=n_sites,
        manifest={
            "input_cif": str(cif_path.resolve()),
            "prepared": False,
            "supercell_size": list(manual_supercell),
            "atom_swaps": manual_swaps,
            "ratio": manual_ratio,
            "ratio_source": "manual",
            "lattice_element": manual_lattice,
            "species_counts": counts,
        },
    )


def write_prepared_outputs(
    prepared: OptimizationInput,
    out_dir: Path,
    source_cif: Path,
) -> None:
    """Write prepared.cif, manifest.json, and a ready-to-run config.toml."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "prepared.cif", prepared.reference)
    (out_dir / "manifest.json").write_text(json.dumps(prepared.manifest, indent=2), encoding="utf-8")

    config_text = f'''cif_path = "{(out_dir / "prepared.cif").resolve()}"
supercell_size = [1, 1, 1]

[prep_params]
auto_prepare = false

[relax_params]
calc_name = "TensorNet-MatPES-PBE-2025.2"
optimizer = "BFGS"
interval = 1
fmax = 0.2
max_steps = 500
relax_atoms = true
relax_cell = true
cell_filter = "FrechetCellFilter"
fix_symmetry = false
fix_atoms = false
symprec = 1.0e-3

[agent_params]
backend = "openai"
model = "gpt-4o-mini"
max_iters = 20
max_swaps_per_step = 4
seed = 0
skill_path = "Skill.md"
out_dir = "runs/agent"

[ga_params]
num_gens = 400
pop_size = 200
random_seed = 0
out_dir = "runs/ga"

[mc_params]
num_steps = 100
acceptance_rate = 0.3
seed = 0
out_dir = "runs/mc"
'''
    (out_dir / "config.toml").write_text(config_text, encoding="utf-8")
    print(f"Prepared structure written to {out_dir / 'prepared.cif'}")
    print(f"Run optimizers with: python agentic_optimizer.py --config {out_dir / 'config.toml'}")


def write_input_manifest(out_dir: Path, manifest: dict) -> None:
    """Copy prep metadata into the run directory for reproducibility."""
    if manifest:
        (out_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def print_loading_info(opt_in: OptimizationInput, cfg: dict) -> None:
    """Standard startup lines shared by agent, GA, and MC."""
    print(f"Loading structure from {opt_in.manifest.get('input_cif', cfg.get('cif_path'))}")
    if opt_in.prepared:
        print(f"Using prepared supercell (min_cell_size={opt_in.min_cell_size})")
    else:
        print(f"Using supercell {opt_in.supercell_size}")
