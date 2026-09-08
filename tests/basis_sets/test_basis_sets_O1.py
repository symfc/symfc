"""Tests of FCBasisSetO1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from symfc.basis_sets import FCBasisSetO1
from symfc.utils.utils import SymfcAtoms

cwd = Path(__file__).parent


def test_fc_basis_set_o1():
    """Test symmetry adapted basis sets of FCBasisSetO1."""
    lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    numbers = [1, 1]
    supercell = SymfcAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)
    sbs = FCBasisSetO1(supercell, log_level=1)
    try:
        sbs.run()
    except ValueError:
        assert sbs.full_basis_set is None


def test_fc_basis_set_o1_nacl222(cell_nacl_222: SymfcAtoms):
    """Test symmetry adapted basis sets of FCBasisSetO1 by nacl222."""
    sbs = FCBasisSetO1(cell_nacl_222, log_level=1)
    try:
        sbs.run()
    except ValueError:
        assert sbs.full_basis_set is None


def test_fc_basis_set_o1_wurtzite332(cell_wurtzite_332: SymfcAtoms):
    """Test symmetry adapted basis sets of FCBasisSetO1 by wurtzite332."""
    sbs = FCBasisSetO1(cell_wurtzite_332, log_level=1).run()
    basis = sbs.full_basis_set.toarray()
    assert basis.shape[0] == 216
    assert basis.shape[1] == 1
    assert np.linalg.norm(basis) ** 2 == pytest.approx(1)
