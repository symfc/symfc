"""Tests of FCBasisSetO2."""

from pathlib import Path

import numpy as np
import phonopy
import pytest
from phonopy import Phonopy

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


def test_fc2_NaCl_222(ph_nacl_222: Phonopy):
    """Test force constants by NaCl 64 atoms supercell and compared with ALM.

    This test uses FCBasisSetO2.

    Also test force constants by NaCl 64 atoms supercell.

    This test with ALM is skipped when ALM is not installed.

    """
    basis_set = FCBasisSetO2(ph_nacl_222.supercell, log_level=1).run()
    ph = phonopy.load(cwd / ".." / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    fc_solver = FCSolverO2(basis_set, log_level=1).solve(
        ph.dataset["displacements"], ph.dataset["forces"]
    )
    fc_compact = fc_solver.compact_fc
    ph_ref = phonopy.load(
        cwd / ".." / "phonopy_NaCl_222_fc.yaml.xz",
        produce_fc=False,
    )
    np.testing.assert_allclose(ph_ref.force_constants, fc_compact, atol=1e-6)


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_NaCl_222_wrt_ALM(ph_nacl_222: Phonopy, is_compact_fc: bool):
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_NaCl_222_rd.yaml.xz",
        FCBasisSetO2(ph_nacl_222.supercell, log_level=1),
        is_compact_fc=is_compact_fc,
    )


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_SnO2_223_wrt_ALM(ph_sno2_223: Phonopy, is_compact_fc: bool):
    """Test force constants by SnO2 72 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SnO2_223_rd.yaml.xz",
        FCBasisSetO2(ph_sno2_223.supercell, log_level=1),
        is_compact_fc=is_compact_fc,
    )


def test_fc2_SnO2_222_wrt_ALM(ph_sno2_222: Phonopy):
    """Test force constants by SnO2 48 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SnO2_222_rd.yaml.xz",
        FCBasisSetO2(ph_sno2_222.supercell, log_level=1),
    )


def test_fc2_SiO2_222_wrt_ALM(ph_sio2_222: Phonopy):
    """Test force constants by SiO2 72 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SiO2_222_rd.yaml.xz",
        FCBasisSetO2(ph_sio2_222.supercell, log_level=1),
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_SiO2_222_fc.yaml", "phonopy_SiO2_222_rd.yaml.xz", fc_compact
    # )


def test_fc2_SiO2_221_wrt_ALM(ph_sio2_221: Phonopy):
    """Test force constants by SiO2 36 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SiO2_221_rd.yaml.xz",
        FCBasisSetO2(ph_sio2_221.supercell, log_level=1),
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_SiO2_221_fc.yaml", "phonopy_SiO2_221_rd.yaml.xz", fc_compact
    # )


def test_fc2_GaN_442_wrt_ALM(ph_gan_442: Phonopy):
    """Test force constants by GaN 128 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_GaN_442_rd.yaml.xz",
        FCBasisSetO2(ph_gan_442.supercell, log_level=1),
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_GaN_442_fc.yaml", "phonopy_GaN_442_rd.yaml.xz", fc_compact
    # )


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_GaN_222_wrt_ALM(ph_gan_222: Phonopy, is_compact_fc: bool):
    """Test force constants by GaN 32 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_GaN_222_rd.yaml.xz",
        FCBasisSetO2(ph_gan_222.supercell, log_level=1),
        is_compact_fc=is_compact_fc,
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_GaN_222_fc.yaml", "phonopy_GaN_222_rd.yaml.xz", fc_compact
    # )


def _compare_fc2_with_alm(
    filename: Path,
    fc_basis_set: FCBasisSetO2,
    is_compact_fc: bool = True,
) -> np.ndarray:
    pytest.importorskip("alm")
    basis_set = fc_basis_set.run()
    ph = phonopy.load(filename, fc_calculator="alm", is_compact_fc=is_compact_fc)
    fc_solver = FCSolverO2(basis_set, log_level=1).solve(
        ph.dataset["displacements"], ph.dataset["forces"]
    )
    if is_compact_fc:
        fc = fc_solver.compact_fc
    else:
        fc = fc_solver.full_fc
    np.testing.assert_allclose(ph.force_constants, fc, atol=1e-6)
    return fc
