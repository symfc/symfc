"""Tests of FCBasisSetBase."""

import pytest
from phonopy import Phonopy

from symfc.basis_sets import FCBasisSetBase


def test_base_fc_basis_set(ph_nacl_222: Phonopy):
    """Test that FCBasisSet can not be instantiate."""
    with pytest.raises(TypeError):
        _ = FCBasisSetBase(ph_nacl_222.supercell)
