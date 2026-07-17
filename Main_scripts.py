import matcalc as mtc
from ase.io import read,write

import ase
from packaging import version
if version.parse(ase.__version__) > version.parse("3.22.1"):
    from ase.filters import FrechetCellFilter as CellFilter
else:
    from ase.constraints import FrechetCellFilter as CellFilter


def setup_optimizer(calc_name="TensorNet-MatPES-PBE-2025.2",
                    optimizer="BFGS",
                    interval=1,
                    fmax=0.2,
                    max_steps=500,
                    relax_atoms=True,
                    relax_cell=True,
                    cell_filter=CellFilter, # hydrostrain is by defult false
                    fix_symmetry=False,
                    fix_atoms=False,
                    symprec=1e-3,
                    ):

    """
    Setup the optimizer for the structure optimization.
    Args:
        calc_name: The name of the calculator to use.
        optimizer: The optimizer to use.
        interval: The interval of the optimizer.
        fmax: The maximum force to use.
        max_steps: The maximum number of steps to use.
        relax_atoms: Whether to relax the atoms.
        relax_cell: Whether to relax the cell.
        cell_filter: The cell filter to use.
        fix_symmetry: Whether to fix the symmetry.
        fix_atoms: Whether to fix the atoms.
        symprec: The symmetry precision to use.
    """

    calculator = mtc.load_fp(calc_name)
    relax_calc = mtc.RelaxCalc(
                calculator,
                optimizer=optimizer,
                traj_file='relaxation.traj',
                interval=interval,
                fmax=fmax,
                max_steps=max_steps,
                relax_atoms=relax_atoms,
                relax_cell=relax_cell,
                cell_filter=cell_filter, # hydrostrain is by defult false
                fix_symmetry=fix_symmetry,
                fix_atoms=fix_atoms,
                symprec=symprec,
                )

    return relax_calc

def get_reference_structures(cif_path,supercell_size=(1,1,1)):
    """
    Get the reference structures for the structure optimization.
    Args:
        cif_path: The path to the CIF file.
        super_cell_size: The size of the super cell.
    """
    reference_structure = read(cif_path)*supercell_size
    return reference_structure

def get_score(structure,optimizer):
    """
    Optimize structure and get the total energy as the score.
    Args:
        structure: The structure to get the score of.
    """
    results = optimizer.calc(structure)
    structure_optimized = results['final_structure']
    return results['energy']

