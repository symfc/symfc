"""Tests of FCBasisSetO3."""

from pathlib import Path

import numpy as np
import pytest
from symfc.basis_sets import FCBasisSetO4
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_fc_basis_set_o4():
    """Test symmetry adapted basis sets of FCBasisSetO4."""
    a = 5.4335600299999998
    lattice = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
    positions = np.array(
        [
            [0.875, 0.875, 0.875],
            [0.875, 0.375, 0.375],
            [0.375, 0.875, 0.375],
            [0.375, 0.375, 0.875],
            [0.125, 0.125, 0.125],
            [0.125, 0.625, 0.625],
            [0.625, 0.125, 0.625],
            [0.625, 0.625, 0.125],
        ]
    )
    numbers = [1, 1, 1, 1, 1, 1, 1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    sbs = FCBasisSetO4(supercell, log_level=1).run()

    assert sbs.basis_set.shape[0] == 115
    assert sbs.basis_set.shape[1] == 72
    compact_basis = sbs.compact_compression_matrix @ sbs.basis_set
    assert np.linalg.norm(compact_basis) ** 2 == pytest.approx(18.0)

    sbs = FCBasisSetO4(supercell, cutoff=3.5, log_level=1).run()
    assert sbs.basis_set.shape[0] == 11
    assert sbs.basis_set.shape[1] == 2
    compact_basis = sbs.compact_compression_matrix @ sbs.basis_set
    assert np.linalg.norm(compact_basis) ** 2 == pytest.approx(0.5)


def test_fc_basis_set_o4_wurtzite():
    """Test symmetry adapted basis sets of FCBasisSetO4."""
    lattice = np.array(
        [
            [3.786186160293827, 0, 0],
            [-1.893093080146913, 3.278933398271515, 0],
            [0, 0, 6.212678269409001],
        ]
    )
    positions = np.array(
        [
            [0.333333333333333, 0.666666666666667, 0.002126465711614],
            [0.666666666666667, 0.333333333333333, 0.502126465711614],
            [0.333333333333333, 0.666666666666667, 0.376316514288389],
            [0.666666666666667, 0.333333333333333, 0.876316514288389],
        ]
    )
    numbers = [1, 1, 2, 2]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    sbs = FCBasisSetO4(supercell, log_level=1).run()

    assert sbs.basis_set.shape[0] == 120
    assert sbs.basis_set.shape[1] == 34
    compact_basis = sbs.compact_compression_matrix @ sbs.basis_set
    assert np.linalg.norm(compact_basis) ** 2 == pytest.approx(34.0)

    sbs = FCBasisSetO4(supercell, cutoff=3.0, log_level=1).run()
    assert sbs.basis_set.shape[0] == 34
    assert sbs.basis_set.shape[1] == 2
    compact_basis = sbs.compact_compression_matrix @ sbs.basis_set
    assert np.linalg.norm(compact_basis) ** 2 == pytest.approx(2.0)


def test_fc_basis_set_o3_wurtzite_221():
    """Test symmetry adapted basis sets of FCBasisSetO3."""
    lattice = np.array(
        [
            [7.572372320587654, 0.0, 0.0],
            [-3.786186160293826, 6.55786679654303, 0.0],
            [0.0, 0.0, 6.212678269409001],
        ]
    )
    positions = np.array(
        [
            [0.166666666666666, 0.333333333333333, 0.002126465711614],
            [0.666666666666667, 0.333333333333333, 0.002126465711614],
            [0.166666666666666, 0.833333333333333, 0.002126465711614],
            [0.666666666666667, 0.833333333333333, 0.002126465711614],
            [0.333333333333333, 0.166666666666666, 0.502126465711614],
            [0.833333333333333, 0.166666666666666, 0.502126465711614],
            [0.333333333333333, 0.666666666666667, 0.502126465711614],
            [0.833333333333333, 0.666666666666667, 0.502126465711614],
            [0.166666666666666, 0.333333333333333, 0.376316514288389],
            [0.666666666666667, 0.333333333333333, 0.376316514288389],
            [0.166666666666666, 0.833333333333333, 0.376316514288389],
            [0.666666666666667, 0.833333333333333, 0.376316514288389],
            [0.333333333333333, 0.166666666666666, 0.876316514288389],
            [0.833333333333333, 0.166666666666666, 0.876316514288389],
            [0.333333333333333, 0.666666666666667, 0.876316514288389],
            [0.833333333333333, 0.666666666666667, 0.876316514288389],
        ]
    )
    numbers = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

    """
    sbs = FCBasisSetO4(supercell, cutoff=4.5, log_level=1).run()
    assert sbs.basis_set.shape[0] == 3733
    assert sbs.basis_set.shape[1] == 2749
    """
    sbs = FCBasisSetO4(supercell, cutoff=4.0, log_level=1).run()
    assert sbs.basis_set.shape[0] == 948
    assert sbs.basis_set.shape[1] == 515

    sbs = FCBasisSetO4(supercell, cutoff=3.5, log_level=1).run()
    assert sbs.basis_set.shape[0] == 61
    assert sbs.basis_set.shape[1] == 4
