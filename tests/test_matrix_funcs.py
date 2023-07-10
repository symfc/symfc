"""Tests of matrix manipulating functions."""

import numpy as np
import pytest
from phonopy import Phonopy
from scipy.sparse import csr_array

from symfc.matrix_funcs import kron_c
from symfc.spg_reps import SpgReps


@pytest.mark.parametrize(
    "pure_translation_only,rank_result",
    [(True, 1152), (False, 42)],
)
def test_kron_c_NaCl_222(
    ph_nacl_222: Phonopy, pure_translation_only: bool, rank_result: int
):
    """Test kron_c by its rank."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        pure_translation_only=pure_translation_only,
        log_level=1,
    )
    natom = len(ph.supercell)
    size_sq = natom**2 * 9
    row, col, data = kron_c(sym_op_reps.representations, natom)
    proj_mat = csr_array((data, (row, col)), shape=(size_sq, size_sq), dtype="double")
    rank = np.rint(proj_mat.diagonal().sum()).astype(int)
    assert rank == rank_result
