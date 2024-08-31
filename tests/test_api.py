"""Tests of Symfc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symfc import Symfc
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_api_NaCl_222(ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray]):
    """Test Symfc class."""
    supercell, displacements, forces = ph_nacl_222
    symfc = Symfc(supercell)
    symfc.compute_basis_set(2)
    symfc.displacements = displacements
    np.testing.assert_array_almost_equal(symfc.displacements, displacements)
    symfc.forces = forces
    np.testing.assert_array_almost_equal(symfc.forces, forces)
    symfc.solve(2)
    fc = symfc.force_constants[2]
    fc_ref = np.loadtxt(cwd / "compact_fc_NaCl_222.xz").reshape(fc.shape)
    np.testing.assert_allclose(fc, fc_ref)


def test_api_NaCl_222_with_dataset(
    ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray],
):
    """Test Symfc class with displacements and forces as input."""
    supercell, displacements, forces = ph_nacl_222
    symfc = Symfc(
        supercell,
        displacements=displacements,
        forces=forces,
    ).run(max_order=2)
    fc = symfc.force_constants[2]
    fc_ref = np.loadtxt(cwd / "compact_fc_NaCl_222.xz").reshape(fc.shape)
    np.testing.assert_allclose(fc, fc_ref)


def test_api_NaCl_222_exception(ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray]):
    """Test Symfc class with displacements and forces as input."""
    supercell, _, _ = ph_nacl_222
    symfc = Symfc(supercell)
    symfc.compute_basis_set(2)
    with pytest.raises(RuntimeError):
        symfc.solve(2)


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_api_si_111_fc3(
    ph3_si_111_fc3: tuple[SymfcAtoms, np.ndarray, np.ndarray], is_compact_fc: bool
):
    """Test Symfc class with displacements and forces as input."""
    supercell, displacements, forces = ph3_si_111_fc3
    symfc = Symfc(supercell, displacements=displacements, forces=forces).run(
        max_order=3, is_compact_fc=is_compact_fc
    )
    fc2 = symfc.force_constants[2]
    fc3 = symfc.force_constants[3]

    if is_compact_fc:
        # np.savetxt(cwd / "compact_fc_Si_111_fc3_2.xz", fc2.ravel())
        # np.savetxt(cwd / "compact_fc_Si_111_fc3_3.xz", fc3.ravel())
        fc2_ref = np.loadtxt(cwd / "compact_fc_Si_111_fc3_2.xz").reshape(fc2.shape)
        fc3_ref = np.loadtxt(cwd / "compact_fc_Si_111_fc3_3.xz").reshape(fc3.shape)
    else:
        # np.savetxt(cwd / "full_fc_Si_111_fc3_2.xz", fc2.ravel())
        # np.savetxt(cwd / "full_fc_Si_111_fc3_3.xz", fc3.ravel())
        fc2_ref = np.loadtxt(cwd / "full_fc_Si_111_fc3_2.xz").reshape(fc2.shape)
        fc3_ref = np.loadtxt(cwd / "full_fc_Si_111_fc3_3.xz").reshape(fc3.shape)
    np.testing.assert_allclose(fc2_ref, fc2, atol=1e-6)
    np.testing.assert_allclose(fc3_ref, fc3, atol=1e-6)
