"""Tests of matrix manipulating functions."""

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgReps
from symfc.utils import (
    _get_perm_compr_matrix_reference,
    get_atomic_lat_trans_decompr_indices,
    get_indep_atoms_by_lat_trans,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
    get_perm_compr_matrix,
)


def test_get_indep_atoms_by_lattice_translation(ph_nacl_222: Phonopy):
    """Test of get_indep_atoms_by_lattice_translation."""
    ph = ph_nacl_222
    sym_op_reps = SpgReps(ph.supercell)
    trans_perms = sym_op_reps.translation_permutations
    assert trans_perms.shape == (32, 64)
    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    np.testing.assert_array_equal(indep_atoms, [0, 32])


def test_get_perm_compr_matrix():
    """Test of get_perm_compr_matrix."""
    C1 = get_perm_compr_matrix(8)
    C2 = _get_perm_compr_matrix_reference(8)
    np.testing.assert_array_almost_equal((C1 @ C1.T).toarray(), (C2 @ C2.T).toarray())
    np.testing.assert_array_almost_equal((C1.T @ C1).toarray(), (C2.T @ C2).toarray())


def test_get_lat_trans_decompr_indices(cell_nacl_111: PhonopyAtoms):
    """Test of get_lat_trans_decompr_indices.

    The one dimensional array with row-size of compr-mat.
    Every element indicates column position.

    """
    ref = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
    ]
    spg_reps = SpgReps(cell_nacl_111)
    trans_perms = spg_reps.translation_permutations
    assert trans_perms.shape == (4, 8)
    decompr_idx = get_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(ref, decompr_idx)


def test_get_lat_trans_compr_indices(cell_nacl_111: PhonopyAtoms):
    """Test get_lat_trans_compr_indices.

    The two dimensional array (n_a * N * 9, n_lp) stores NN33 indices where
    compression matrix elements are non-zero.

    """
    spg_reps = SpgReps(cell_nacl_111)
    trans_perms = spg_reps.translation_permutations
    n_lp, N = trans_perms.shape
    assert trans_perms.shape == (4, 8)
    decompr_mat = get_lat_trans_decompr_indices(trans_perms)
    compr_mat = get_lat_trans_compr_matrix(decompr_mat, N, n_lp).toarray()
    compr_idx = get_lat_trans_compr_indices(trans_perms)
    for c, elem_idx in enumerate(compr_idx):
        for r in elem_idx:
            np.testing.assert_almost_equal(compr_mat[r, c], 0.5)


def test_get_atomic_lat_trans_decompr_indices(cell_nacl_111: PhonopyAtoms):
    """Test of get_atomic_lat_trans_decompr_indices.

    This function is an atomic version of get_lat_trans_decompr_indices.
    The one dimensional array with row-size of compr-mat.
    Every element indicates column position.

    """
    ref = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        1,
        0,
        3,
        2,
        5,
        4,
        7,
        6,
        2,
        3,
        0,
        1,
        6,
        7,
        4,
        5,
        3,
        2,
        1,
        0,
        7,
        6,
        5,
        4,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        9,
        8,
        11,
        10,
        13,
        12,
        15,
        14,
        10,
        11,
        8,
        9,
        14,
        15,
        12,
        13,
        11,
        10,
        9,
        8,
        15,
        14,
        13,
        12,
    ]
    spg_reps = SpgReps(cell_nacl_111)
    trans_perms = spg_reps.translation_permutations
    assert trans_perms.shape == (4, 8)
    atomic_decompr_idx = get_atomic_lat_trans_decompr_indices(trans_perms)
    np.testing.assert_array_equal(ref, atomic_decompr_idx)
