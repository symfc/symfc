"""Tests of combination tools."""

from __future__ import annotations

from symfc.utils.combination_tools import (
    _get_combinations_indep_atoms,
    _get_entire_combinations,
    get_combinations,
    get_combinations_first_atom,
)
from symfc.utils.cutoff_tools import FCCutoff


def test_get_entire_combinations_local():
    """Test _get_entire_combinations."""
    combs = _get_entire_combinations(5, 2)
    assert combs.shape == (10, 2)
    combs = _get_entire_combinations(6, 3)
    assert combs.shape == (20, 3)


def test_get_combinations_indep_atoms():
    """Test _get_combinations_indep_atoms."""
    combs = _get_combinations_indep_atoms(9, 2, [0, 2])
    assert combs.shape == (24, 2)
    combs = _get_combinations_indep_atoms(9, 3, [0, 1])
    assert combs.shape == (83, 3)


def test_get_combinations():
    """Test get_combinations."""
    combs = get_combinations(4, order=2, indep_atoms=[0, 2])
    assert combs.shape == (42, 2)
    combs = get_combinations(4, order=3, indep_atoms=[0, 2])
    assert combs.shape == (155, 3)
    combs = get_combinations(4, order=4, indep_atoms=[0, 2])
    assert combs.shape == (384, 4)

    combs = get_combinations(4, order=2, indep_atoms=[0, 1, 2, 3])
    assert combs.shape == (66, 2)
    combs = get_combinations(4, order=2)
    assert combs.shape == (66, 2)


def test_get_combinations_cutoff(ph_gan_222):
    """Test get_combinations with cutoff distance."""
    supercell, _, _ = ph_gan_222
    fc_cutoff = FCCutoff(supercell, cutoff=3.0)
    combs = get_combinations(len(supercell.numbers), order=2, fc_cutoff=fc_cutoff)
    assert combs.shape == (672, 2)
    combs = get_combinations(len(supercell.numbers), order=3, fc_cutoff=fc_cutoff)
    assert combs.shape == (1184, 3)

    fc_cutoff = FCCutoff(supercell, cutoff=2.5)
    combs = get_combinations(len(supercell.numbers), order=4, fc_cutoff=fc_cutoff)
    assert combs.shape == (960, 4)


def test_get_combinations_first_atom():
    """Test get_combinations with first atom."""
    combs = list(get_combinations_first_atom(4, order=2, first_atom=0))
    assert combs[0].shape == (11, 2)
    assert combs[1].shape == (10, 2)
    assert combs[2].shape == (9, 2)

    combs = list(get_combinations_first_atom(4, order=3, first_atom=0))
    assert combs[0].shape == (55, 3)
    assert combs[1].shape == (45, 3)
    assert combs[2].shape == (36, 3)
    combs = list(get_combinations_first_atom(4, order=4, first_atom=0))
    assert combs[0].shape == (165, 4)
    assert combs[1].shape == (120, 4)
    assert combs[2].shape == (84, 4)
    combs = list(get_combinations_first_atom(4, order=2, first_atom=2))
    assert combs[0].shape == (5, 2)
    assert combs[1].shape == (4, 2)
    assert combs[2].shape == (3, 2)
    combs = list(get_combinations_first_atom(4, order=3, first_atom=2))
    assert combs[0].shape == (10, 3)
    assert combs[1].shape == (6, 3)
    assert combs[2].shape == (3, 3)
    combs = list(get_combinations_first_atom(4, order=4, first_atom=2))
    assert combs[0].shape == (10, 4)
    assert combs[1].shape == (4, 4)
    assert combs[2].shape == (1, 4)


def test_get_combinations_first_atom_cutoff(ph_gan_222):
    """Test get_combinations with first atom."""
    supercell, _, _ = ph_gan_222
    fc_cutoff = FCCutoff(supercell, cutoff=3.0)

    combs = list(
        get_combinations_first_atom(
            len(supercell.numbers), order=3, first_atom=0, fc_cutoff=fc_cutoff
        )
    )
    assert combs[0].shape == (37, 3)
    assert combs[1].shape == (24, 3)
    assert combs[2].shape == (12, 3)

    combs = list(
        get_combinations_first_atom(
            len(supercell.numbers), order=4, first_atom=0, fc_cutoff=fc_cutoff
        )
    )
    assert combs[0].shape == (40, 4)
    assert combs[1].shape == (16, 4)
    assert combs[2].shape == (4, 4)
