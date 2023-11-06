"""Tests of Symfc."""
from symfc import Symfc
from phonopy import Phonopy
import numpy as np
import pytest


def test_api_NaCl_222(ph_nacl_222: Phonopy):
    """Test Symfc class."""
    pytest.importorskip("alm")
    ph = ph_nacl_222
    symfc = Symfc(ph.supercell)
    symfc.displacements = ph.displacements
    np.testing.assert_array_almost_equal(symfc.displacements, ph.displacements)
    symfc.forces = ph.forces
    np.testing.assert_array_almost_equal(symfc.forces, ph.forces)
    symfc.calculate_basis_set()
    symfc.solve()
    fc = symfc.force_constants
    ph.produce_force_constants(
        fc_calculator="alm", calculate_full_force_constants=False
    )
    np.testing.assert_array_almost_equal(fc, ph.force_constants)


def test_api_NaCl_222_with_dataset(ph_nacl_222: Phonopy):
    """Test Symfc class with displacements and forces as input."""
    pytest.importorskip("alm")
    ph = ph_nacl_222
    symfc = Symfc(ph.supercell, displacements=ph.displacements, forces=ph.forces)
    fc = symfc.force_constants
    ph.produce_force_constants(
        fc_calculator="alm", calculate_full_force_constants=False
    )
    np.testing.assert_array_almost_equal(fc, ph.force_constants)
