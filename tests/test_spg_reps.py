"""Tests of SpgReps class."""

import numpy as np
import pytest
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps


@pytest.mark.parametrize(
    "pure_translation_only,rank_result",
    [(True, 6), (False, 0)],
)
def test_SpgReps_NaCl_111(
    cell_nacl_111: PhonopyAtoms, pure_translation_only: bool, rank_result: int
):
    """Test SpgReps by its rank."""
    cell = cell_nacl_111
    sym_op_reps = SpgReps(
        cell.cell.T,
        cell.scaled_positions.T,
        cell.numbers,
        pure_translation_only=pure_translation_only,
        log_level=1,
    )

    reps = sym_op_reps.representations
    proj = reps[0]
    if len(reps) > 1:
        for rep in reps[1:]:
            proj += rep
    proj /= len(reps)
    for v in proj.toarray():
        print(v)
    rank = np.rint(proj.diagonal().sum()).astype(int)
    assert rank == rank_result
    if pure_translation_only:
        np.testing.assert_allclose(proj.data, 0.25)
    else:
        len(proj.data) == 0


@pytest.mark.parametrize(
    "pure_translation_only,rank_result",
    [(True, 6), (False, 0)],
)
def test_SpgReps_NaCl_222(
    ph_nacl_222: Phonopy, pure_translation_only: bool, rank_result: int
):
    """Test SpgReps by its rank."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(
        ph.supercell.cell.T,
        ph.supercell.scaled_positions.T,
        ph.supercell.numbers,
        pure_translation_only=pure_translation_only,
        log_level=1,
    )

    reps = sym_op_reps.representations
    proj = reps[0]
    if len(reps) > 1:
        for rep in reps[1:]:
            proj += rep
    proj /= len(reps)
    rank = np.rint(proj.diagonal().sum()).astype(int)
    assert rank == rank_result
    if pure_translation_only:
        np.testing.assert_allclose(proj.data, 0.03125)
    else:
        len(proj.data) == 0
