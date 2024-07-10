"""Tests of FC cutoff tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_FCCutoff():
    """Test FCCutoff using NaCl conventional unit cell 2x2x2."""
    lattice = [
        [11.281120000000000, 0.000000000000000, 0.000000000000000],
        [0.000000000000000, 11.281120000000000, 0.000000000000000],
        [0.000000000000000, 0.000000000000000, 11.281120000000000],
    ]
    points = [
        [0.000000000000000, 0.000000000000000, 0.000000000000000],
        [0.500000000000000, 0.000000000000000, 0.000000000000000],
        [0.000000000000000, 0.500000000000000, 0.000000000000000],
        [0.500000000000000, 0.500000000000000, 0.000000000000000],
        [0.000000000000000, 0.000000000000000, 0.500000000000000],
        [0.500000000000000, 0.000000000000000, 0.500000000000000],
        [0.000000000000000, 0.500000000000000, 0.500000000000000],
        [0.500000000000000, 0.500000000000000, 0.500000000000000],
        [0.000000000000000, 0.250000000000000, 0.250000000000000],
        [0.500000000000000, 0.250000000000000, 0.250000000000000],
        [0.000000000000000, 0.750000000000000, 0.250000000000000],
        [0.500000000000000, 0.750000000000000, 0.250000000000000],
        [0.000000000000000, 0.250000000000000, 0.750000000000000],
        [0.500000000000000, 0.250000000000000, 0.750000000000000],
        [0.000000000000000, 0.750000000000000, 0.750000000000000],
        [0.500000000000000, 0.750000000000000, 0.750000000000000],
        [0.250000000000000, 0.000000000000000, 0.250000000000000],
        [0.750000000000000, 0.000000000000000, 0.250000000000000],
        [0.250000000000000, 0.500000000000000, 0.250000000000000],
        [0.750000000000000, 0.500000000000000, 0.250000000000000],
        [0.250000000000000, 0.000000000000000, 0.750000000000000],
        [0.750000000000000, 0.000000000000000, 0.750000000000000],
        [0.250000000000000, 0.500000000000000, 0.750000000000000],
        [0.750000000000000, 0.500000000000000, 0.750000000000000],
        [0.250000000000000, 0.250000000000000, 0.000000000000000],
        [0.750000000000000, 0.250000000000000, 0.000000000000000],
        [0.250000000000000, 0.750000000000000, 0.000000000000000],
        [0.750000000000000, 0.750000000000000, 0.000000000000000],
        [0.250000000000000, 0.250000000000000, 0.500000000000000],
        [0.750000000000000, 0.250000000000000, 0.500000000000000],
        [0.250000000000000, 0.750000000000000, 0.500000000000000],
        [0.750000000000000, 0.750000000000000, 0.500000000000000],
        [0.250000000000000, 0.250000000000000, 0.250000000000000],
        [0.750000000000000, 0.250000000000000, 0.250000000000000],
        [0.250000000000000, 0.750000000000000, 0.250000000000000],
        [0.750000000000000, 0.750000000000000, 0.250000000000000],
        [0.250000000000000, 0.250000000000000, 0.750000000000000],
        [0.750000000000000, 0.250000000000000, 0.750000000000000],
        [0.250000000000000, 0.750000000000000, 0.750000000000000],
        [0.750000000000000, 0.750000000000000, 0.750000000000000],
        [0.250000000000000, 0.000000000000000, 0.000000000000000],
        [0.750000000000000, 0.000000000000000, 0.000000000000000],
        [0.250000000000000, 0.500000000000000, 0.000000000000000],
        [0.750000000000000, 0.500000000000000, 0.000000000000000],
        [0.250000000000000, 0.000000000000000, 0.500000000000000],
        [0.750000000000000, 0.000000000000000, 0.500000000000000],
        [0.250000000000000, 0.500000000000000, 0.500000000000000],
        [0.750000000000000, 0.500000000000000, 0.500000000000000],
        [0.000000000000000, 0.250000000000000, 0.000000000000000],
        [0.500000000000000, 0.250000000000000, 0.000000000000000],
        [0.000000000000000, 0.750000000000000, 0.000000000000000],
        [0.500000000000000, 0.750000000000000, 0.000000000000000],
        [0.000000000000000, 0.250000000000000, 0.500000000000000],
        [0.500000000000000, 0.250000000000000, 0.500000000000000],
        [0.000000000000000, 0.750000000000000, 0.500000000000000],
        [0.500000000000000, 0.750000000000000, 0.500000000000000],
        [0.000000000000000, 0.000000000000000, 0.250000000000000],
        [0.500000000000000, 0.000000000000000, 0.250000000000000],
        [0.000000000000000, 0.500000000000000, 0.250000000000000],
        [0.500000000000000, 0.500000000000000, 0.250000000000000],
        [0.000000000000000, 0.000000000000000, 0.750000000000000],
        [0.500000000000000, 0.000000000000000, 0.750000000000000],
        [0.000000000000000, 0.500000000000000, 0.750000000000000],
        [0.500000000000000, 0.500000000000000, 0.750000000000000],
    ]
    numbers = [
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
        17,
    ]
    n_nonzero_elem = _check(lattice, points, numbers)

    ref = [11200, 32896, 185536, 262144]
    np.testing.assert_equal(ref, n_nonzero_elem)


def test_FCCutoff_gan_443():
    """Test FCCutoff using GaN 4x4x3.

    import spglib
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.cells import get_supercell

    dataset = spglib.get_symmetry_dataset(
        (cell_gan_111.cell, cell_gan_111.scaled_positions, cell_gan_111.numbers)
    )
    phonopy_cell = PhonopyAtoms(
        cell=dataset["std_lattice"],
        scaled_positions=dataset["std_positions"],
        numbers=dataset["std_types"],
    )
    scell = get_supercell(phonopy_cell, [[4, 0, 0], [0, 4, 0], [0, 0, 3]])
    for v in scell.cell:
        print(f"[{v[0]:.10f}, {v[1]:.10f}, {v[2]:.10f}],")
    for v in scell.scaled_positions:
        print(f"[{v[0]:.10f}, {v[1]:.10f}, {v[2]:.10f}],")
    print(", ".join([f"{n}" for n in scell.numbers]))

    """
    lattice = [
        [12.7230620734, 0.0000000000, 0.0000000000],
        [-6.3615310367, 11.0184949695, 0.0000000000],
        [0.0000000000, 0.0000000000, 15.5478175200],
    ]
    points = [
        [0.0000000000, 0.0000000000, 15.5478175200],
        [0.0833333333, 0.1666666667, 0.0413973067],
        [0.3333333333, 0.1666666667, 0.0413973067],
        [0.5833333333, 0.1666666667, 0.0413973067],
        [0.8333333333, 0.1666666667, 0.0413973067],
        [0.0833333333, 0.4166666667, 0.0413973067],
        [0.3333333333, 0.4166666667, 0.0413973067],
        [0.5833333333, 0.4166666667, 0.0413973067],
        [0.8333333333, 0.4166666667, 0.0413973067],
        [0.0833333333, 0.6666666667, 0.0413973067],
        [0.3333333333, 0.6666666667, 0.0413973067],
        [0.5833333333, 0.6666666667, 0.0413973067],
        [0.8333333333, 0.6666666667, 0.0413973067],
        [0.0833333333, 0.9166666667, 0.0413973067],
        [0.3333333333, 0.9166666667, 0.0413973067],
        [0.5833333333, 0.9166666667, 0.0413973067],
        [0.8333333333, 0.9166666667, 0.0413973067],
        [0.0833333333, 0.1666666667, 0.3747306400],
        [0.3333333333, 0.1666666667, 0.3747306400],
        [0.5833333333, 0.1666666667, 0.3747306400],
        [0.8333333333, 0.1666666667, 0.3747306400],
        [0.0833333333, 0.4166666667, 0.3747306400],
        [0.3333333333, 0.4166666667, 0.3747306400],
        [0.5833333333, 0.4166666667, 0.3747306400],
        [0.8333333333, 0.4166666667, 0.3747306400],
        [0.0833333333, 0.6666666667, 0.3747306400],
        [0.3333333333, 0.6666666667, 0.3747306400],
        [0.5833333333, 0.6666666667, 0.3747306400],
        [0.8333333333, 0.6666666667, 0.3747306400],
        [0.0833333333, 0.9166666667, 0.3747306400],
        [0.3333333333, 0.9166666667, 0.3747306400],
        [0.5833333333, 0.9166666667, 0.3747306400],
        [0.8333333333, 0.9166666667, 0.3747306400],
        [0.0833333333, 0.1666666667, 0.7080639733],
        [0.3333333333, 0.1666666667, 0.7080639733],
        [0.5833333333, 0.1666666667, 0.7080639733],
        [0.8333333333, 0.1666666667, 0.7080639733],
        [0.0833333333, 0.4166666667, 0.7080639733],
        [0.3333333333, 0.4166666667, 0.7080639733],
        [0.5833333333, 0.4166666667, 0.7080639733],
        [0.8333333333, 0.4166666667, 0.7080639733],
        [0.0833333333, 0.6666666667, 0.7080639733],
        [0.3333333333, 0.6666666667, 0.7080639733],
        [0.5833333333, 0.6666666667, 0.7080639733],
        [0.8333333333, 0.6666666667, 0.7080639733],
        [0.0833333333, 0.9166666667, 0.7080639733],
        [0.3333333333, 0.9166666667, 0.7080639733],
        [0.5833333333, 0.9166666667, 0.7080639733],
        [0.8333333333, 0.9166666667, 0.7080639733],
        [0.1666666667, 0.0833333333, 0.2080639733],
        [0.4166666667, 0.0833333333, 0.2080639733],
        [0.6666666667, 0.0833333333, 0.2080639733],
        [0.9166666667, 0.0833333333, 0.2080639733],
        [0.1666666667, 0.3333333333, 0.2080639733],
        [0.4166666667, 0.3333333333, 0.2080639733],
        [0.6666666667, 0.3333333333, 0.2080639733],
        [0.9166666667, 0.3333333333, 0.2080639733],
        [0.1666666667, 0.5833333333, 0.2080639733],
        [0.4166666667, 0.5833333333, 0.2080639733],
        [0.6666666667, 0.5833333333, 0.2080639733],
        [0.9166666667, 0.5833333333, 0.2080639733],
        [0.1666666667, 0.8333333333, 0.2080639733],
        [0.4166666667, 0.8333333333, 0.2080639733],
        [0.6666666667, 0.8333333333, 0.2080639733],
        [0.9166666667, 0.8333333333, 0.2080639733],
        [0.1666666667, 0.0833333333, 0.5413973067],
        [0.4166666667, 0.0833333333, 0.5413973067],
        [0.6666666667, 0.0833333333, 0.5413973067],
        [0.9166666667, 0.0833333333, 0.5413973067],
        [0.1666666667, 0.3333333333, 0.5413973067],
        [0.4166666667, 0.3333333333, 0.5413973067],
        [0.6666666667, 0.3333333333, 0.5413973067],
        [0.9166666667, 0.3333333333, 0.5413973067],
        [0.1666666667, 0.5833333333, 0.5413973067],
        [0.4166666667, 0.5833333333, 0.5413973067],
        [0.6666666667, 0.5833333333, 0.5413973067],
        [0.9166666667, 0.5833333333, 0.5413973067],
        [0.1666666667, 0.8333333333, 0.5413973067],
        [0.4166666667, 0.8333333333, 0.5413973067],
        [0.6666666667, 0.8333333333, 0.5413973067],
        [0.9166666667, 0.8333333333, 0.5413973067],
        [0.1666666667, 0.0833333333, 0.8747306400],
        [0.4166666667, 0.0833333333, 0.8747306400],
        [0.6666666667, 0.0833333333, 0.8747306400],
        [0.9166666667, 0.0833333333, 0.8747306400],
        [0.1666666667, 0.3333333333, 0.8747306400],
        [0.4166666667, 0.3333333333, 0.8747306400],
        [0.6666666667, 0.3333333333, 0.8747306400],
        [0.9166666667, 0.3333333333, 0.8747306400],
        [0.1666666667, 0.5833333333, 0.8747306400],
        [0.4166666667, 0.5833333333, 0.8747306400],
        [0.6666666667, 0.5833333333, 0.8747306400],
        [0.9166666667, 0.5833333333, 0.8747306400],
        [0.1666666667, 0.8333333333, 0.8747306400],
        [0.4166666667, 0.8333333333, 0.8747306400],
        [0.6666666667, 0.8333333333, 0.8747306400],
        [0.9166666667, 0.8333333333, 0.8747306400],
        [0.0833333333, 0.1666666667, 0.1669360267],
        [0.3333333333, 0.1666666667, 0.1669360267],
        [0.5833333333, 0.1666666667, 0.1669360267],
        [0.8333333333, 0.1666666667, 0.1669360267],
        [0.0833333333, 0.4166666667, 0.1669360267],
        [0.3333333333, 0.4166666667, 0.1669360267],
        [0.5833333333, 0.4166666667, 0.1669360267],
        [0.8333333333, 0.4166666667, 0.1669360267],
        [0.0833333333, 0.6666666667, 0.1669360267],
        [0.3333333333, 0.6666666667, 0.1669360267],
        [0.5833333333, 0.6666666667, 0.1669360267],
        [0.8333333333, 0.6666666667, 0.1669360267],
        [0.0833333333, 0.9166666667, 0.1669360267],
        [0.3333333333, 0.9166666667, 0.1669360267],
        [0.5833333333, 0.9166666667, 0.1669360267],
        [0.8333333333, 0.9166666667, 0.1669360267],
        [0.0833333333, 0.1666666667, 0.5002693600],
        [0.3333333333, 0.1666666667, 0.5002693600],
        [0.5833333333, 0.1666666667, 0.5002693600],
        [0.8333333333, 0.1666666667, 0.5002693600],
        [0.0833333333, 0.4166666667, 0.5002693600],
        [0.3333333333, 0.4166666667, 0.5002693600],
        [0.5833333333, 0.4166666667, 0.5002693600],
        [0.8333333333, 0.4166666667, 0.5002693600],
        [0.0833333333, 0.6666666667, 0.5002693600],
        [0.3333333333, 0.6666666667, 0.5002693600],
        [0.5833333333, 0.6666666667, 0.5002693600],
        [0.8333333333, 0.6666666667, 0.5002693600],
        [0.0833333333, 0.9166666667, 0.5002693600],
        [0.3333333333, 0.9166666667, 0.5002693600],
        [0.5833333333, 0.9166666667, 0.5002693600],
        [0.8333333333, 0.9166666667, 0.5002693600],
        [0.0833333333, 0.1666666667, 0.8336026933],
        [0.3333333333, 0.1666666667, 0.8336026933],
        [0.5833333333, 0.1666666667, 0.8336026933],
        [0.8333333333, 0.1666666667, 0.8336026933],
        [0.0833333333, 0.4166666667, 0.8336026933],
        [0.3333333333, 0.4166666667, 0.8336026933],
        [0.5833333333, 0.4166666667, 0.8336026933],
        [0.8333333333, 0.4166666667, 0.8336026933],
        [0.0833333333, 0.6666666667, 0.8336026933],
        [0.3333333333, 0.6666666667, 0.8336026933],
        [0.5833333333, 0.6666666667, 0.8336026933],
        [0.8333333333, 0.6666666667, 0.8336026933],
        [0.0833333333, 0.9166666667, 0.8336026933],
        [0.3333333333, 0.9166666667, 0.8336026933],
        [0.5833333333, 0.9166666667, 0.8336026933],
        [0.8333333333, 0.9166666667, 0.8336026933],
        [0.1666666667, 0.0833333333, 0.0002693600],
        [0.4166666667, 0.0833333333, 0.0002693600],
        [0.6666666667, 0.0833333333, 0.0002693600],
        [0.9166666667, 0.0833333333, 0.0002693600],
        [0.1666666667, 0.3333333333, 0.0002693600],
        [0.4166666667, 0.3333333333, 0.0002693600],
        [0.6666666667, 0.3333333333, 0.0002693600],
        [0.9166666667, 0.3333333333, 0.0002693600],
        [0.1666666667, 0.5833333333, 0.0002693600],
        [0.4166666667, 0.5833333333, 0.0002693600],
        [0.6666666667, 0.5833333333, 0.0002693600],
        [0.9166666667, 0.5833333333, 0.0002693600],
        [0.1666666667, 0.8333333333, 0.0002693600],
        [0.4166666667, 0.8333333333, 0.0002693600],
        [0.6666666667, 0.8333333333, 0.0002693600],
        [0.9166666667, 0.8333333333, 0.0002693600],
        [0.1666666667, 0.0833333333, 0.3336026933],
        [0.4166666667, 0.0833333333, 0.3336026933],
        [0.6666666667, 0.0833333333, 0.3336026933],
        [0.9166666667, 0.0833333333, 0.3336026933],
        [0.1666666667, 0.3333333333, 0.3336026933],
        [0.4166666667, 0.3333333333, 0.3336026933],
        [0.6666666667, 0.3333333333, 0.3336026933],
        [0.9166666667, 0.3333333333, 0.3336026933],
        [0.1666666667, 0.5833333333, 0.3336026933],
        [0.4166666667, 0.5833333333, 0.3336026933],
        [0.6666666667, 0.5833333333, 0.3336026933],
        [0.9166666667, 0.5833333333, 0.3336026933],
        [0.1666666667, 0.8333333333, 0.3336026933],
        [0.4166666667, 0.8333333333, 0.3336026933],
        [0.6666666667, 0.8333333333, 0.3336026933],
        [0.9166666667, 0.8333333333, 0.3336026933],
        [0.1666666667, 0.0833333333, 0.6669360267],
        [0.4166666667, 0.0833333333, 0.6669360267],
        [0.6666666667, 0.0833333333, 0.6669360267],
        [0.9166666667, 0.0833333333, 0.6669360267],
        [0.1666666667, 0.3333333333, 0.6669360267],
        [0.4166666667, 0.3333333333, 0.6669360267],
        [0.6666666667, 0.3333333333, 0.6669360267],
        [0.9166666667, 0.3333333333, 0.6669360267],
        [0.1666666667, 0.5833333333, 0.6669360267],
        [0.4166666667, 0.5833333333, 0.6669360267],
        [0.6666666667, 0.5833333333, 0.6669360267],
        [0.9166666667, 0.5833333333, 0.6669360267],
        [0.1666666667, 0.8333333333, 0.6669360267],
        [0.4166666667, 0.8333333333, 0.6669360267],
        [0.6666666667, 0.8333333333, 0.6669360267],
        [0.9166666667, 0.8333333333, 0.6669360267],
    ]
    numbers = [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
    ]

    n_nonzero_elem = _check(lattice, points, numbers)
    print(n_nonzero_elem)
    ref = [70303, 676837, 3900487, 6748021]
    np.testing.assert_equal(ref, n_nonzero_elem)


def _check(lattice: np.ndarray, points: np.ndarray, numbers: np.ndarray):
    n_nonzero_elem = []
    for cutoff in (4, 6, 8, 10):
        fccutoff = FCCutoff(
            SymfcAtoms(cell=lattice, scaled_positions=points, numbers=numbers),
            cutoff=cutoff,
        )
        nonzero_idx = [
            f"{i}"
            for i, nonzero in enumerate(fccutoff.nonzero_atomic_indices_fc3())
            if nonzero
        ]
        n_nonzero_elem.append(len(nonzero_idx))

    return n_nonzero_elem
