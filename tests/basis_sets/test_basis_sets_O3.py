"""Tests of FCBasisSetO3."""

from pathlib import Path

import numpy as np
import phono3py
import pytest
from phono3py import Phono3py

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO3
from symfc.solvers import FCSolverO2O3
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O3 import get_lat_trans_compr_matrix_O3

cwd = Path(__file__).parent


def test_fc_basis_set_o3():
    """Test symmetry adapted basis sets of FCBasisSetO3."""
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
    sbs = FCBasisSetO3(supercell, log_level=1).run()

    basis_ref = [
        0.671875,
        0.656250,
        0.765625,
        1.000000,
        0.750000,
        0.875000,
        0.562500,
        0.765625,
        0.718750,
        0.671875,
        1.000000,
        0.875000,
        0.875000,
        0.625000,
        0.750000,
        0.562500,
        0.875000,
    ]
    np.testing.assert_allclose(
        np.sort([v @ v for v in sbs.basis_set]), np.sort(basis_ref)
    )

    assert np.linalg.norm(sbs.basis_set) ** 2 == pytest.approx(13.0)

    np.testing.assert_array_equal(
        sbs.compact_compression_matrix.tocoo().row[[0, 6, 9]], [5, 32, 42]
    )
    np.testing.assert_allclose(
        sbs.compression_matrix.data[[0, 6, 9]],
        [-0.1443375672974064, 0.058925565098878946, 0.0833333333333333],
    )

    np.testing.assert_array_equal(
        sbs.compression_matrix.tocoo().row[[0, 6, 9]], [5, 32, 42]
    )
    np.testing.assert_allclose(
        sbs.compression_matrix.data[[0, 6, 9]],
        [-0.1443375672974064, 0.058925565098878946, 0.0833333333333333],
    )

    lat_trans_compr_matrix_O3 = get_lat_trans_compr_matrix_O3(
        sbs.translation_permutations
    )
    np.testing.assert_allclose(lat_trans_compr_matrix_O3.data, [0.5] * 13824)
    assert lat_trans_compr_matrix_O3.indices[-1] == 2726


def test_si_111_222(ph3_si_111_222: Phono3py):
    """Test fc2 and fc3 by Si-111-222 supercells and compared with ALM.

    This test with ALM is skipped when ALM is not installed.

    """
    ph3 = ph3_si_111_222
    basis_set_o2 = FCBasisSetO2(ph3.supercell, log_level=1).run()
    basis_set_o3 = FCBasisSetO3(ph3.supercell, log_level=1).run()
    ph3 = phono3py.load(
        cwd / ".." / "phono3py_params_Si-111-222-rd.yaml.xz", produce_fc=False
    )
    d = ph3.dataset["displacements"]
    f = ph3.dataset["forces"]
    ph3 = Phono3py(
        ph3.unitcell,
        supercell_matrix=ph3.supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
    )
    ph3.displacements = d
    ph3.forces = f
    ph3.produce_fc3(fc_calculator="alm", is_compact_fc=True)
    fc_solver = FCSolverO2O3([basis_set_o2, basis_set_o3], log_level=1).solve(d, f)
    fc2, fc3 = fc_solver.compact_fc
    np.testing.assert_allclose(ph3.fc2, fc2, atol=1e-6)
    np.testing.assert_allclose(ph3.fc3, fc3, atol=1e-6)
