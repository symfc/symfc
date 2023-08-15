"""Tests of FCBasisSets."""
from pathlib import Path

import numpy as np
import phonopy
import pytest

from symfc.fc_basis import FCBasisSets
from symfc.spg_reps import SpgReps

cwd = Path(__file__).parent


@pytest.mark.parametrize(
    "lang,use_exact_projection_matrix",
    [("Py", False), ("Py_for_C", False), ("C", False), ("C", True)],
)
def test_fc_basis_sets(lang, use_exact_projection_matrix):
    """Test symmetry adapted basis sets of FC."""
    basis_ref = [
        [-0.28867513, 0, 0, 0.28867513, 0, 0],
        [0, -0.28867513, 0, 0, 0.28867513, 0],
        [0, 0, -0.28867513, 0, 0, 0.28867513],
        [0.28867513, 0, 0, -0.28867513, 0, 0],
        [0, 0.28867513, 0, 0, -0.28867513, 0],
        [0, 0, 0.28867513, 0, 0, -0.28867513],
    ]

    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]).T
    types = [0, 0]

    sym_op_reps = SpgReps(lattice, positions, types)
    rep = sym_op_reps.representations
    sbs = FCBasisSets(
        rep,
        use_exact_projection_matrix=use_exact_projection_matrix,
        log_level=1,
        lang=lang,
    )
    basis = sbs.basis_sets_matrix_form
    np.testing.assert_allclose(basis[0], basis_ref, atol=1e-6)
    assert np.linalg.norm(basis[0]) == pytest.approx(1.0)


def test_fc_NaCl_222(bs_nacl_222: np.ndarray):
    """Test force constants by NaCl 64 atoms supercell and compared with ALM.

    Also test force constants by NaCl 64 atoms supercell.

    This test with ALM is skipped when ALM is not installed.

    """
    basis_sets = bs_nacl_222
    ph = phonopy.load(cwd / "phonopy_NaCl_222_rd.yaml.xz", produce_fc=False)
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

    Bf = np.einsum("ijklm,nkm->injl", basis_sets, d).reshape(basis_sets.shape[0], -1)
    c = -f.ravel() @ np.linalg.pinv(Bf)
    fc = np.einsum("i,ijklm->jklm", c, basis_sets)
    fc_compact = fc[ph.primitive.p2s_map]

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

    ph_ref = phonopy.load(cwd / "phonopy_NaCl_222_fc.yaml.xz", produce_fc=False)
    np.testing.assert_allclose(ph_ref.force_constants, fc_compact, atol=1e-6)

    _ = _compare_fc_with_alm("phonopy_NaCl_222_rd.yaml.xz", bs_nacl_222)


@pytest.mark.big
def test_fc_SnO2_223_wrt_ALM(bs_sno2_223: np.ndarray):
    """Test force constants by SnO2 72 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc_with_alm("phonopy_SnO2_223_rd.yaml.xz", bs_sno2_223)


def test_fc_SnO2_222_wrt_ALM(bs_sno2_222: np.ndarray):
    """Test force constants by SnO2 48 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc_with_alm("phonopy_SnO2_222_rd.yaml.xz", bs_sno2_222)


@pytest.mark.big
def test_fc_SiO2_222_wrt_ALM(bs_sio2_222: np.ndarray):
    """Test force constants by SiO2 72 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc_with_alm("phonopy_SiO2_222_rd.yaml.xz", bs_sio2_222)
    # _write_phonopy_fc_yaml(
    #     "phonopy_SiO2_222_fc.yaml", "phonopy_SiO2_222_rd.yaml.xz", fc_compact
    # )


def test_fc_SiO2_221_wrt_ALM(bs_sio2_221: np.ndarray):
    """Test force constants by SiO2 36 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc_with_alm("phonopy_SiO2_221_rd.yaml.xz", bs_sio2_221)
    # _write_phonopy_fc_yaml(
    #     "phonopy_SiO2_221_fc.yaml", "phonopy_SiO2_221_rd.yaml.xz", fc_compact
    # )


@pytest.mark.big
def test_fc_GaN_442_wrt_ALM(bs_gan_442: np.ndarray):
    """Test force constants by GaN 128 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc_with_alm("phonopy_GaN_442_rd.yaml.xz", bs_gan_442)
    # _write_phonopy_fc_yaml(
    #     "phonopy_GaN_442_fc.yaml", "phonopy_GaN_442_rd.yaml.xz", fc_compact
    # )


def test_fc_GaN_222_wrt_ALM(bs_gan_222: np.ndarray):
    """Test force constants by GaN 32 atoms supercell and compared with ALM.

    This test is skipped when ALM is not installed.

    """
    _ = _compare_fc_with_alm("phonopy_GaN_222_rd.yaml.xz", bs_gan_222)
    # _write_phonopy_fc_yaml(
    #     "phonopy_GaN_222_fc.yaml", "phonopy_GaN_222_rd.yaml.xz", fc_compact
    # )


def _compare_fc_with_alm(filename, basis_sets) -> np.ndarray:
    pytest.importorskip("alm")
    ph = phonopy.load(cwd / filename, fc_calculator="alm")
    f = ph.dataset["forces"]
    d = ph.dataset["displacements"]
    Bf = np.einsum("ijklm,nkm->injl", basis_sets, d).reshape(basis_sets.shape[0], -1)
    c = -f.ravel() @ np.linalg.pinv(Bf)
    fc = np.einsum("i,ijklm->jklm", c, basis_sets)
    fc_compact = fc[ph.primitive.p2s_map]
    np.testing.assert_allclose(ph.force_constants, fc_compact, atol=1e-6)
    return fc_compact


def _write_phonopy_fc_yaml(output_filename, input_filename, fc_compact):
    ph = phonopy.load(cwd / input_filename, produce_fc=False)
    ph.force_constants = fc_compact
    save_settings = {
        "force_sets": False,
        "displacements": False,
        "force_constants": True,
    }
    ph.save(output_filename, settings=save_settings)
