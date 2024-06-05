"""Tests of Symfc."""

import numpy as np
import pytest
from phonopy import Phonopy

from symfc import Symfc


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
