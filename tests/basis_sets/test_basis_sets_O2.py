"""Tests of FCBasisSetO2."""

from pathlib import Path
from typing import Union

import numpy as np
import phonopy
import pytest
from phonopy import Phonopy

from symfc.basis_sets import FCBasisSetO2, FCBasisSetO2Slow
from symfc.solvers import FCSolverO2
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_fc_basis_set_o2():
    """Test symmetry adapted basis sets of FCBasisSetO2."""
    basis_ref = [
        [-0.28867513, 0, 0, 0.28867513, 0, 0],
        [0, -0.28867513, 0, 0, 0.28867513, 0],
        [0, 0, -0.28867513, 0, 0, 0.28867513],
        [0.28867513, 0, 0, -0.28867513, 0, 0],
        [0, 0.28867513, 0, 0, -0.28867513, 0],
        [0, 0, 0.28867513, 0, 0, -0.28867513],
    ]

    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    sbs = FCBasisSetO2(supercell, log_level=1).run()
    N = len(supercell)
    basis = np.transpose(sbs.full_basis_set.reshape(N, N, 3, 3), (0, 2, 1, 3)).reshape(
        N * 3, N * 3
    )
    np.testing.assert_allclose(basis, basis_ref, atol=1e-6)
    assert np.linalg.norm(basis) == pytest.approx(1.0)


def test_fc_basis_set_o2_slow():
    """Test symmetry adapted basis sets of FCBasisSetO2Slow."""
    basis_ref = [
        [-0.28867513, 0, 0, 0.28867513, 0, 0],
        [0, -0.28867513, 0, 0, 0.28867513, 0],
        [0, 0, -0.28867513, 0, 0, 0.28867513],
        [0.28867513, 0, 0, -0.28867513, 0, 0],
        [0, 0.28867513, 0, 0, -0.28867513, 0],
        [0, 0, 0.28867513, 0, 0, -0.28867513],
    ]
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    sbs = FCBasisSetO2Slow(supercell, log_level=1).run()

    def _convert_basis_set_o2_slow_matrix_form(
        basis_set_o2: FCBasisSetO2Slow,
    ) -> list[np.ndarray]:
        """Convert basis set to matrix form (n_bases, 3N, 3N)."""
        trans_perms = basis_set_o2.translation_permutations
        N = trans_perms.shape[1]
        decompr_idx = np.transpose(
            basis_set_o2.decompression_indices.reshape(N, N, 3, 3), (0, 2, 1, 3)
        ).reshape(N * 3, N * 3)
        b_mat_all = []
        for b in basis_set_o2.basis_set.T:
            b_mat_all.append(b[decompr_idx] / np.sqrt(trans_perms.shape[0]))
        return b_mat_all

    basis = _convert_basis_set_o2_slow_matrix_form(sbs)
    np.testing.assert_allclose(basis[0], basis_ref, atol=1e-6)
    assert np.linalg.norm(basis[0]) == pytest.approx(1.0)


def test_fc2_NaCl_222(ph_nacl_222: Phonopy):
    """Test force constants by NaCl 64 atoms supercell and compared with ALM.

    This test uses FCBasisSetO2.

    Also test force constants by NaCl 64 atoms supercell.

    This test with ALM is skipped when ALM is not installed.

    """
    basis_set = FCBasisSetO2(ph_nacl_222.supercell, log_level=1).run()
    ph = phonopy.load(cwd / ".." / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    f = ph.dataset["forces"]
    d = ph.dataset["displacements"]
    solver = FCSolverO2(basis_set, log_level=1)
    fc_compact = solver.solve(d, f, is_compact_fc=True)
    ph_ref = phonopy.load(
        cwd / ".." / "phonopy_NaCl_222_fc.yaml.xz",
        produce_fc=False,
    )
    np.testing.assert_allclose(ph_ref.force_constants, fc_compact, atol=1e-6)


def test_fc2_slow_NaCl_222(ph_nacl_222: Phonopy):
    """Test force constants by NaCl 64 atoms supercell and compared with ALM.

    This test uses FCBasisSetO2Slow.

    Also test force constants by NaCl 64 atoms supercell.

    This test with ALM is skipped when ALM is not installed.

    """
    basis_set = FCBasisSetO2Slow(ph_nacl_222.supercell, log_level=1).run()
    ph = phonopy.load(cwd / ".." / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
    f = ph.dataset["forces"]
    d = ph.dataset["displacements"]

    # To save and load basis_sets.
    # np.savetxt(
    #     "basis_sets_NaCl.dat",
    #     basis_sets.reshape((basis_sets.shape[0], -1)),
    # )
    # Data size of basis_sets_NaCl in bz2 is ~6.8M.
    # This is too large to store in git for distribution.
    # So below is just an example and for convenience in writing test.
    # basis_sets = np.loadtxt(cwd / "basis_sets_NaCl.dat.bz2", dtype="double").reshape(
    #     (31, 64, 64, 3, 3)
    # )

    solver = FCSolverO2(basis_set, log_level=1)
    fc_compact = solver.solve(d, f)

    # To save force constants in phonopy-yaml.
    # save_settings = {
    #     "force_constants": True,
    #     "displacements": False,
    #     "force_sets": False,
    # }
    # ph.save(
    #     "phonopy_NaCl222_fc.yaml",
    #     settings=save_settings
    # )

    ph_ref = phonopy.load(cwd / ".." / "phonopy_NaCl_222_fc.yaml.xz", produce_fc=False)
    np.testing.assert_allclose(ph_ref.force_constants, fc_compact, atol=1e-6)

    # def _write_phonopy_fc_yaml(output_filename, input_filename, fc_compact):
    #     ph = phonopy.load(input_filename, produce_fc=False)
    #     ph.force_constants = fc_compact
    #     save_settings = {
    #         "force_sets": False,
    #         "displacements": False,
    #         "force_constants": True,
    #     }
    #     ph.save(output_filename, settings=save_settings)

    # _write_phonopy_fc_yaml(
    #     "phonopy_NaCl_222_fc.yaml", "phonopy_NaCl_222_rd.yaml.xz", fc_compact
    # )


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_slow_NaCl_222_wrt_ALM(ph_nacl_222: Phonopy, is_compact_fc: bool):
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_NaCl_222_rd.yaml.xz",
        FCBasisSetO2Slow(ph_nacl_222.supercell, log_level=1),
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


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_slow_SnO2_223_wrt_ALM(ph_sno2_223: Phonopy, is_compact_fc: bool):
    """Test force constants by SnO2 72 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SnO2_223_rd.yaml.xz",
        FCBasisSetO2Slow(ph_sno2_223.supercell, log_level=1),
        is_compact_fc=is_compact_fc,
    )


def test_fc2_slow_SnO2_222_wrt_ALM(ph_sno2_222: Phonopy):
    """Test force constants by SnO2 48 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SnO2_222_rd.yaml.xz",
        FCBasisSetO2Slow(ph_sno2_222.supercell, log_level=1),
    )


@pytest.mark.big
def test_fc2_slow_SiO2_222_wrt_ALM(ph_sio2_222: Phonopy):
    """Test force constants by SiO2 72 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SiO2_222_rd.yaml.xz",
        FCBasisSetO2Slow(ph_sio2_222.supercell, log_level=1),
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_SiO2_222_fc.yaml", "phonopy_SiO2_222_rd.yaml.xz", fc_compact
    # )


def test_fc2_slow_SiO2_221_wrt_ALM(ph_sio2_221: Phonopy):
    """Test force constants by SiO2 36 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_SiO2_221_rd.yaml.xz",
        FCBasisSetO2Slow(ph_sio2_221.supercell, log_level=1),
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_SiO2_221_fc.yaml", "phonopy_SiO2_221_rd.yaml.xz", fc_compact
    # )


def test_fc2_slow_GaN_442_wrt_ALM(ph_gan_442: Phonopy):
    """Test force constants by GaN 128 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_GaN_442_rd.yaml.xz",
        FCBasisSetO2Slow(ph_gan_442.supercell, log_level=1),
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_GaN_442_fc.yaml", "phonopy_GaN_442_rd.yaml.xz", fc_compact
    # )


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_fc2_slow_GaN_222_wrt_ALM(ph_gan_222: Phonopy, is_compact_fc: bool):
    """Test force constants by GaN 32 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc2_with_alm(
        cwd / ".." / "phonopy_GaN_222_rd.yaml.xz",
        FCBasisSetO2Slow(ph_gan_222.supercell, log_level=1),
        is_compact_fc=is_compact_fc,
    )
    # _write_phonopy_fc_yaml(
    #     "phonopy_GaN_222_fc.yaml", "phonopy_GaN_222_rd.yaml.xz", fc_compact
    # )


def test_full_basis_set_o2_slow_NaCl222_wrt_ALM(ph_nacl_222: Phonopy):
    _ = _full_basis_set_o2_compare_with_alm(
        cwd / ".." / "phonopy_NaCl_222_rd.yaml.xz",
        FCBasisSetO2Slow(ph_nacl_222.supercell, log_level=1),
    )


def test_full_basis_set_o2_slow_SnO2_223_wrt_ALM(ph_sno2_223: Phonopy):
    _ = _full_basis_set_o2_compare_with_alm(
        cwd / ".." / "phonopy_SnO2_223_rd.yaml.xz",
        FCBasisSetO2Slow(ph_sno2_223.supercell, log_level=1),
    )


def _compare_fc2_with_alm(
    filename: Path,
    fc_basis_set: Union[FCBasisSetO2, FCBasisSetO2Slow],
    is_compact_fc: bool = True,
) -> np.ndarray:
    pytest.importorskip("alm")
    basis_set = fc_basis_set.run()
    ph = phonopy.load(filename, fc_calculator="alm", is_compact_fc=is_compact_fc)
    f = ph.dataset["forces"]
    d = ph.dataset["displacements"]
    solver = FCSolverO2(basis_set, log_level=1)
    fc_compact = solver.solve(d, f, is_compact_fc=is_compact_fc)
    np.testing.assert_allclose(ph.force_constants, fc_compact, atol=1e-6)
    return fc_compact


def _full_basis_set_o2_compare_with_alm(
    filename: Path,
    fc_basis_set: Union[FCBasisSetO2Slow, FCBasisSetO2],
):
    """This is the test for FCBasisSetO2.full_basis_set."""
    pytest.importorskip("alm")
    basis_set = fc_basis_set.run()
    ph = phonopy.load(filename, fc_calculator="alm", is_compact_fc=False)
    f = ph.dataset["forces"]
    d = ph.dataset["displacements"]
    full_basis_set = basis_set.full_basis_set
    n_bases = full_basis_set.shape[-1]
    N = len(ph.supercell)
    square_basis_set = full_basis_set.reshape(N, N, 3, 3, n_bases)
    coeff = (
        -np.linalg.pinv(
            np.einsum("ijk,jlkmn->ilmn", d, square_basis_set).reshape(-1, n_bases)
        )
        @ f.ravel()
    )
    full_fc = (full_basis_set @ coeff).reshape(N, N, 3, 3)
    np.testing.assert_allclose(ph.force_constants, full_fc, atol=1e-6)
