"""Combination utility functions."""

from typing import Optional

import numpy as np

from symfc.utils.cutoff_tools import FCCutoff

SCIPY_SPARSE_DATA_LIMIT = 2147483647


def _get_entire_combinations(n: int, r: int):
    """Return numpy array of combinations.

    combinations = np.array(
       list(itertools.combinations(range(n), r)), dtype=int
    )
    """
    combs = np.ones((r, n - r + 1), dtype="int_")
    combs[0] = np.arange(n - r + 1)
    for j in range(1, r):
        reps = (n - r + j) - combs[j - 1]
        combs = np.repeat(combs, reps, axis=1)
        ind = np.add.accumulate(reps)
        combs[j, ind[:-1]] = 1 - reps[1:]
        combs[j, 0] = j
        combs[j] = np.add.accumulate(combs[j])
    return combs.T


def _get_combinations_first_atom(n: int, r: int, first_atom: int):
    """Return numpy array of combinations related to independent atoms."""
    first_indices = [first_atom * 3 + i for i in range(3)]
    # combs_all = []
    for first in first_indices:
        combs = _get_entire_combinations(n - first, r - 1) + first
        out = np.empty((combs.shape[0], r), dtype=combs.dtype)
        out[:, 0] = first
        out[:, 1:] = combs
        yield out
        # combs_all.append(out)
    # return np.vstack(combs_all)


def _get_combinations_indep_atoms(n: int, r: int, indep_atoms: list):
    """Return numpy array of combinations related to independent atoms."""
    first_indices = [i * 3 + j for i in indep_atoms for j in range(3)]
    combs_all = []
    for first in first_indices:
        combs = _get_entire_combinations(n - first, r - 1) + first
        out = np.empty((combs.shape[0], r), dtype=combs.dtype)
        out[:, 0] = first
        out[:, 1:] = combs
        combs_all.append(out)
    return np.vstack(combs_all)


def get_combinations_first_atom(
    natom: int,
    order: int,
    first_atom: int,
    fc_cutoff: Optional[FCCutoff] = None,
    indep_atoms: Optional[np.ndarray] = None,
):
    """Return numpy array of FC index combinations."""
    if fc_cutoff is None:
        return _get_combinations_first_atom(3 * natom, order, first_atom)


#    if fc_cutoff is not None:
#        if order == 2:
#            combinations = fc_cutoff.combinations2()
#        elif order == 3:
#            """Combinations can be divided using fc_cut.combiations3(i)."""
#            combinations = fc_cutoff.combinations3_all()
#        elif order == 4:
#            combinations = fc_cutoff.combinations4_all()
#        else:
#            raise NotImplementedError(
#                "Combinations are implemented only for 2 <= order <= 4."
#            )
#    else:
#        combinations = _get_entire_combinations(3 * natom, order)
#
#    if indep_atoms is not None:
#        nonzero = np.zeros(combinations.shape[0], dtype=bool)
#        atom_indices = combinations[:, 0] // 3
#        for i in indep_atoms:
#            nonzero[atom_indices == i] = True
#        combinations = combinations[nonzero]
#    return combinations


def get_combinations(
    natom: int,
    order: int,
    fc_cutoff: Optional[FCCutoff] = None,
    indep_atoms: Optional[np.ndarray] = None,
):
    """Return numpy array of FC index combinations."""
    if fc_cutoff is None and indep_atoms is None:
        return _get_entire_combinations(3 * natom, order)

    if fc_cutoff is None and len(indep_atoms) < natom:
        return _get_combinations_indep_atoms(3 * natom, order, indep_atoms)

    if fc_cutoff is not None:
        if order == 2:
            combinations = fc_cutoff.combinations2()
        elif order == 3:
            """Combinations can be divided using fc_cut.combiations3(i)."""
            combinations = fc_cutoff.combinations3_all()
        elif order == 4:
            combinations = fc_cutoff.combinations4_all()
        else:
            raise NotImplementedError(
                "Combinations are implemented only for 2 <= order <= 4."
            )
    else:
        combinations = _get_entire_combinations(3 * natom, order)

    if indep_atoms is not None:
        nonzero = np.zeros(combinations.shape[0], dtype=bool)
        atom_indices = combinations[:, 0] // 3
        for i in indep_atoms:
            nonzero[atom_indices == i] = True
        combinations = combinations[nonzero]
    return combinations


def get_combinations_stable(
    natom: int,
    order: int,
    fc_cutoff: Optional[FCCutoff] = None,
    indep_atoms: Optional[np.ndarray] = None,
):
    """Return numpy array of FC index combinations."""
    if fc_cutoff is None:
        combinations = _get_entire_combinations(3 * natom, order)
    else:
        if order == 2:
            combinations = fc_cutoff.combinations2()
        elif order == 3:
            """Combinations can be divided using fc_cut.combiations3(i)."""
            combinations = fc_cutoff.combinations3_all()
        elif order == 4:
            combinations = fc_cutoff.combinations4_all()
        else:
            raise NotImplementedError(
                "Combinations are implemented only for 2 <= order <= 4."
            )

    if indep_atoms is not None:
        nonzero = np.zeros(combinations.shape[0], dtype=bool)
        atom_indices = combinations[:, 0] // 3
        for i in indep_atoms:
            nonzero[atom_indices == i] = True
        combinations = combinations[nonzero]
    return combinations
