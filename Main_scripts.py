import matcalc as mtc
from ase.io import read
from ase import Atoms

import ase
from packaging import version

if version.parse(ase.__version__) > version.parse("3.22.1"):
    from ase.filters import FrechetCellFilter as CellFilter
else:
    from ase.constraints import FrechetCellFilter as CellFilter


CELL_FILTERS = {
    "CellFilter": CellFilter,
    "FrechetCellFilter": CellFilter,
}


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
    """
    Setup the optimizer for the structure optimization.
    """
    if isinstance(cell_filter, str):
        cell_filter = CELL_FILTERS.get(cell_filter, CellFilter)

    calculator = mtc.load_fp(calc_name)
    relax_calc = mtc.RelaxCalc(
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
    return relax_calc


def get_reference_structures(cif_path, supercell_size=(1, 1, 1)):
    """
    Get the reference supercell from a CIF file.
    """
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
    """
    Return a copy of lattice sites with symbols set from ``occupancy``.
    """
    if len(occupancy) != len(template_sites):
        raise ValueError(
            f"Occupancy length {len(occupancy)} != number of sites {len(template_sites)}"
        )
    sites = template_sites.copy()
    for i, symbol in enumerate(occupancy):
        sites[i].symbol = symbol
    return sites


def get_score(structure, optimizer):
    """
    Optimize a structure and return total energy (eV) plus results dict.
    """
    results = optimizer.calc(structure)
    return results["energy"], results
