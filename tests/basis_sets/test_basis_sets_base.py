"""Tests of FCBasisSetBase."""

from __future__ import annotations

import numpy as np
import pytest
from symfc.basis_sets import FCBasisSetBase
from symfc.utils.utils import SymfcAtoms


def test_base_fc_basis_set(ph_nacl_222: tuple[SymfcAtoms, np.ndarray, np.ndarray]):
    """Test that FCBasisSet can not be instantiate."""
    with pytest.raises(TypeError):
        _ = FCBasisSetBase(ph_nacl_222[0])
