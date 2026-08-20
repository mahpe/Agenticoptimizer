"""
Prepare a disordered CIF for the Agenticoptimizer scripts.

Builds the smallest supercell with integer swap-site occupancies, writes
``prepared.cif`` + ``config.toml``, and can be consumed by all three optimizers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import (
    identify_disorder_sublattices,
    normalize_supercell_size,
    parse_cif_structure,
    prepare_disordered_input,
    write_prepared_outputs,
)

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare a disordered CIF for agentic / GA / MC optimizers."
    )
    p.add_argument(
        "--input-cif",
        type=Path,
        required=True,
        help="Disordered input CIF",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for prepared.cif, manifest.json, config.toml",
    )
    p.add_argument(
        "--max-cell-size",
        type=int,
        default=20,
        help="Maximum scalar supercell factor to try",
    )
    p.add_argument(
        "--sublattice-index",
        type=int,
        default=0,
        help="Which detected disorder group to expand (0 = first)",
    )
    p.add_argument(
        "--occupancy-tolerance",
        type=float,
        default=2.0,
        help="pymatgen CIF occupancy tolerance",
    )
    p.add_argument(
        "--lattice-element",
        type=str,
        default=None,
        help="Placeholder element on swap sites (default: dominant species)",
    )
    p.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=None,
        metavar="FRAC",
        help="Manual species fractions (must sum to 1; requires --atom-swaps)",
    )
    p.add_argument(
        "--atom-swaps",
        type=str,
        nargs="+",
        default=None,
        help="Manual swap species (default: read from CIF)",
    )
    p.add_argument(
        "--supercell-size",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Manual supercell (1 int or 3 ints, e.g. 2 or 2 2 2)",
    )
    p.add_argument(
        "--list-sublattices",
        action="store_true",
        help="Only list detected disorder groups and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_cif = args.input_cif.resolve()
    out_dir = args.out_dir.resolve()

    if args.list_sublattices:
        structure = parse_cif_structure(input_cif, occupancy_tolerance=args.occupancy_tolerance)
        sublattices = identify_disorder_sublattices(structure)
        if not sublattices:
            print("No disordered sites detected.")
            return 0
        for index, sub in enumerate(sublattices):
            print(f"[{index}] {sub['label']}  ({len(sub['site_indices'])} sites)")
        return 0

    supercell_size = (
        normalize_supercell_size(args.supercell_size) if args.supercell_size else None
    )
    if args.ratio is not None and abs(sum(args.ratio) - 1.0) > 1e-6:
        print("Error: --ratio must sum to 1.", file=sys.stderr)
        return 1

    prepared = prepare_disordered_input(
        cif_path=input_cif,
        max_cell_size=args.max_cell_size,
        sublattice_index=args.sublattice_index,
        occupancy_tolerance=args.occupancy_tolerance,
        lattice_element=args.lattice_element,
        atom_swaps=args.atom_swaps,
        ratio=args.ratio,
        supercell_size=supercell_size,
    )
    if prepared is None:
        print(f"No partial occupancies found in {input_cif}.")
        print("Use an ordered CIF with manual supercell_size / swap_params in config.toml.")
        return 1

    write_prepared_outputs(prepared, out_dir, input_cif)
    print(f"min_cell_size = {prepared.min_cell_size}")
    print(f"swap sites = {prepared.n_swap_sites}")
    print(f"sublattice = {prepared.sublattice_label}")
    print(f"atom_swaps = {prepared.atom_swaps}")
    print(f"ratio = {prepared.ratio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
