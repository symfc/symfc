"""Tests of FCBasisSetO2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symfc.basis_sets import FCBasisSetO2
from symfc.solvers.solver_O2 import FCSolverO2
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_fc_basis_set_o2():
    """Test symmetry adapted basis sets of FCBasisSetO2."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    sbs = FCBasisSetO2(supercell, log_level=1).run()

    np.testing.assert_allclose(
        np.sort(sbs.basis_set), [[-np.sqrt(2) / 2], [np.sqrt(2) / 2]]
    )

    comp_mat = sbs.compression_matrix
    np.testing.assert_allclose(comp_mat.data, [0.40824829046386313] * comp_mat.size)
    ref_col = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ref_row = [0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 31, 35]
    np.testing.assert_array_equal(comp_mat.tocoo().col, ref_col)
    np.testing.assert_array_equal(comp_mat.tocoo().row, ref_row)

    compact_comp_mat = sbs.compact_compression_matrix
    np.testing.assert_allclose(
        compact_comp_mat.data, [0.40824829046386313] * compact_comp_mat.size
    )
    ref_col = [0, 0, 0, 1, 1, 1]
    ref_row = [0, 4, 8, 9, 13, 17]
    np.testing.assert_array_equal(compact_comp_mat.tocoo().col, ref_col)
    np.testing.assert_array_equal(compact_comp_mat.tocoo().row, ref_row)

    assert np.linalg.norm(sbs.basis_set) == pytest.approx(1.0)


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_NaCl_222(
    ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray], is_compact_fc: bool
):
    """Test force constants by NaCl 64 atoms supercell."""
    _assert_fc(ph_nacl_222, "NaCl_222", is_compact_fc)


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_SnO2_223(
    ph_sno2_223: tuple[SymfcAtoms, np.ndarray, np.ndarray], is_compact_fc: bool
):
    """Test force constants by SnO2 72 atoms supercell."""
    _assert_fc(ph_sno2_223, "SnO2_223", is_compact_fc)


def test_fc2_SiO2_221(ph_sio2_221: tuple[SymfcAtoms, np.ndarray, np.ndarray]):
    """Test force constants by SiO2 36 atoms supercell."""
    _assert_fc(ph_sio2_221, "SiO2_221")


def test_fc2_GaN_442(ph_gan_442: tuple[SymfcAtoms, np.ndarray, np.ndarray]):
    """Test force constants by GaN 128 atoms supercell."""
    _assert_fc(ph_gan_442, "GaN_442")


def test_fc2_GaN_222(ph_gan_222: tuple[SymfcAtoms, np.ndarray, np.ndarray]):
    """Test force constants by GaN 32 atoms supercell."""
    _assert_fc(ph_gan_222, "GaN_222")


def _assert_fc(
    ph: tuple[SymfcAtoms, np.ndarray, np.ndarray], name: str, is_compact_fc: bool = True
):
    supercell, displacements, forces = ph
    basis_set = FCBasisSetO2(supercell, log_level=1).run()
    print(basis_set)
    fc_solver = FCSolverO2(basis_set, log_level=1).solve(displacements, forces)
    if is_compact_fc:
        fc = fc_solver.compact_fc
        # np.savetxt(f"compact_fc_{name}.xz", fc.ravel())
        fc_ref = np.loadtxt(cwd / ".." / f"compact_fc_{name}.xz").reshape(fc.shape)
    else:
        fc = fc_solver.full_fc
        fc_ref = np.loadtxt(cwd / ".." / f"full_fc_{name}.xz").reshape(fc.shape)
    np.testing.assert_allclose(fc, fc_ref, atol=1e-6)
