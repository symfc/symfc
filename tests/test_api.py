"""Tests of Symfc."""

import numpy as np
import pytest
from phonopy import Phonopy

from symfc import Symfc
from symfc.basis_sets import FCBasisSetO2Slow


def test_api_NaCl_222(ph_nacl_222: Phonopy):
    """Test Symfc class."""
    pytest.importorskip("alm")
    ph = ph_nacl_222
    symfc = Symfc(ph.supercell)
    symfc.compute_basis_set(2)
    symfc.displacements = ph.displacements
    np.testing.assert_array_almost_equal(symfc.displacements, ph.displacements)
    symfc.forces = ph.forces
    np.testing.assert_array_almost_equal(symfc.forces, ph.forces)
    symfc.solve(
        [
            2,
        ]
    )
    fc = symfc.force_constants[2]
    ph.produce_force_constants(
        fc_calculator="alm", calculate_full_force_constants=False
    )
    np.testing.assert_array_almost_equal(fc, ph.force_constants)


def test_api_NaCl_222_with_dataset(ph_nacl_222: Phonopy):
    """Test Symfc class with displacements and forces as input."""
    pytest.importorskip("alm")
    ph = ph_nacl_222
    symfc = Symfc(
        ph.supercell,
        displacements=ph.displacements,
        forces=ph.forces,
        orders=[
            2,
        ],
    )
    fc = symfc.force_constants[2]
    ph.produce_force_constants(
        fc_calculator="alm", calculate_full_force_constants=False
    )
    np.testing.assert_array_almost_equal(fc, ph.force_constants)


def test_api_NaCl_222_exception(ph_nacl_222: Phonopy):
    """Test Symfc class with displacements and forces as input."""
    pytest.importorskip("alm")
    ph = ph_nacl_222
    symfc = Symfc(ph.supercell)
    symfc.compute_basis_set(2)
    with pytest.raises(RuntimeError):
        symfc.solve(
            orders=[
                2,
            ]
        )


def test_api_full_basis_set_SnO2_223(ph_sno2_223: Phonopy):
    """Test to obtain full basis set."""
    ph = ph_sno2_223
    symfc = Symfc(ph.supercell, displacements=ph.displacements, forces=ph.forces)
    symfc.compute_basis_set(2)
    basis_set: FCBasisSetO2Slow = symfc.basis_set[2]
    full_basis_set = basis_set.full_basis_set
    compressed_basis_set: np.ndarray = basis_set.basis_set
    N = len(ph.supercell)
    assert full_basis_set.shape == (N**2 * 9, compressed_basis_set.shape[1])
