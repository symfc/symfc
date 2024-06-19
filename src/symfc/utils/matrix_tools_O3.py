"""Matrix utility functions for 3rd order force constants."""

import itertools
import math

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools_O3 import FCCutoffO3
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O3 import (
    get_atomic_lat_trans_decompr_indices_O3,
    get_lat_trans_decompr_indices_O3,
)


def N3N3N3_to_NNN333(combinations_perm: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vec = combinations_perm[:, 0] // 3 * 27 * N**2
    vec += combinations_perm[:, 1] // 3 * 27 * N
    vec += combinations_perm[:, 2] // 3 * 27
    vec += combinations_perm[:, 0] % 3 * 9
    vec += combinations_perm[:, 1] % 3 * 3
    vec += combinations_perm[:, 2] % 3
    return vec


def get_perm_compr_matrix_O3(natom: int) -> csr_array:
    """Return compression matrix by permutation symmetry.

    Matrix shape is (NNN333, (N*3)(N*3+1)(N*3+2)/6).

    """
    NNN333 = 27 * natom**3
    combinations3 = np.array(
        list(itertools.combinations(range(3 * natom), 3)), dtype=int
    )
    combinations2 = np.array(
        list(itertools.combinations(range(3 * natom), 2)), dtype=int
    )
    combinations1 = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)

    n_col3 = combinations3.shape[0]
    n_col2 = combinations2.shape[0] * 2
    n_col1 = combinations1.shape[0]
    n_col = n_col3 + n_col2 + n_col1
    n_data3 = combinations3.shape[0] * 6
    n_data2 = combinations2.shape[0] * 6
    n_data1 = combinations1.shape[0]
    n_data = n_data3 + n_data2 + n_data1

    row = np.zeros(n_data, dtype="int_")
    col = np.zeros(n_data, dtype="int_")
    data = np.zeros(n_data, dtype="double")

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    begin_id, end_id = 0, n_data3
    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    combinations_perm = combinations3[:, perms].reshape((-1, 3))
    row[begin_id:end_id] = N3N3N3_to_NNN333(combinations_perm, natom)
    col[begin_id:end_id] = np.repeat(range(n_col3), 6)
    data[begin_id:end_id] = 1 / math.sqrt(6)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    begin_id = end_id
    end_id = begin_id + n_data2
    perms = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    combinations_perm = combinations2[:, perms].reshape((-1, 3))
    row[begin_id:end_id] = N3N3N3_to_NNN333(combinations_perm, natom)
    col[begin_id:end_id] = np.repeat(range(n_col3, n_col3 + n_col2), 3)
    data[begin_id:end_id] = 1 / math.sqrt(3)

    # (1) for FC3 with single index ia
    begin_id = end_id
    row[begin_id:] = N3N3N3_to_NNN333(combinations1, natom)
    col[begin_id:] = np.array(range(n_col3 + n_col2, n_col))
    data[begin_id:] = 1.0

    return csr_array((data, (row, col)), shape=(NNN333, n_col))


def compressed_complement_projector_sum_rules_algo1(
    compress_mat: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    r"""Return complementary projection matrix for sum rule compressed by C.

    C = compress_mat.

    proj_sum_cplmt = [C.T @ Csum(c)] @ [Csum(c).T @ C]
                   = c_sum_cplmt_compr.T @ c_sum_cplmt_compr
    Matrix shape of proj_sum_cplmt is (C.shape[1], C.shape[1]).
    C.shape[0] must be equal to NNN333.

    Sum rules are given as sums over i: \sum_i \phi_{ia,jb,kc} = 0

    """
    NNN333 = 27 * N**3
    NN333 = 27 * N**2

    row = np.arange(NNN333)
    col = np.tile(range(NN333), N)
    data = np.zeros(NNN333)
    data[:] = 1 / math.sqrt(N)
    c_sum_cplmt = csr_array((data, (row, col)), shape=(NNN333, NN333))

    if use_mkl:
        compress_mat = compress_mat.tocsr()
        c_sum_cplmt = c_sum_cplmt.tocsr()

    # bottleneck part
    c_sum_cplmt_compr = dot_product_sparse(c_sum_cplmt.T, compress_mat, use_mkl=use_mkl)
    proj_sum_cplmt = dot_product_sparse(
        c_sum_cplmt_compr.T, c_sum_cplmt_compr, use_mkl=use_mkl
    )
    # bottleneck part: end
    return proj_sum_cplmt


def compressed_complement_projector_sum_rules(
    compress_mat: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    """Return complementary projection matrix for sum rule compressed by C."""
    return compressed_complement_projector_sum_rules_algo1(
        compress_mat, N, use_mkl=use_mkl
    )


def compressed_projector_sum_rules(
    compress_mat: csr_array, N: int, use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule compressed by C."""
    proj_cplmt = compressed_complement_projector_sum_rules(
        compress_mat, N, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


"""
Memory-efficient updated functions for running FCBasisSetO3.
But they have not been yet implemented in FCBasisSetO3.
"""


def set_complement_sum_rules(trans_perms) -> csr_array:
    """Calculate a decomposition of complementary projector for sum rules.

    It is compressed by C_trans without allocating C_trans.
    P_sum^(c) = C_trans.T @ C_sum^(c) @ C_sum^(c).T @ C_trans
              = [C_sum^(c).T @ C_trans].T @ [C_sum^(c).T @ C_trans]

    Return
    ------
    Product of C_sum^(c).T and C_trans.
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NN27 = natom**2 * 27

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((natom, NN27)).T.reshape(-1)

    row = np.repeat(np.arange(NN27), natom)
    c_sum_cplmt = csr_array(
        (
            np.ones(NNN27, dtype="double"),
            (row, decompr_idx),
        ),
        shape=(NN27, NNN27 // n_lp),
        dtype="double",
    )
    c_sum_cplmt /= n_lp * natom
    return c_sum_cplmt


def compressed_complement_projector_sum_rules_from_compact_compr_mat(
    trans_perms, n_a_compress_mat: csr_array, use_mkl: bool = False
) -> csr_array:
    """Calculate a complementary projector for sum rules.

    This is compressed by C_trans and n_a_compress_mat without
    allocating C_trans. Batch calculations are used to reduce
    the amount of memory allocation.

    Return
    ------
    Compressed projector
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NN27 = natom**2 * 27

    proj_size = n_a_compress_mat.shape[1]
    proj_sum_cplmt = csr_array(
        ([], ([], [])),
        shape=(proj_size, proj_size),
        dtype="double",
    )

    decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
    decompr_idx = decompr_idx.reshape((natom, NN27)).T.reshape(-1)

    n_batch = 9
    if n_batch == 3:
        batch_size_vector = natom**3 * 9
        batch_size_matrix = natom**2 * 9
    elif n_batch == 9:
        batch_size_vector = natom**3 * 3
        batch_size_matrix = natom**2 * 3
    elif n_batch == 27:
        batch_size_vector = natom**3
        batch_size_matrix = natom**2
    elif n_batch == natom:
        batch_size_vector = natom**2 * 27
        batch_size_matrix = natom * 27
    else:
        raise ValueError("n_batch = 9, 27, or N.")

    for begin, end in zip(*get_batch_slice(NNN27, batch_size_vector)):
        print("Proj_complement (sum.T @ trans) batch:", end)
        c_sum_cplmt = csr_array(
            (
                np.ones(batch_size_vector, dtype="double"),
                (
                    np.repeat(np.arange(batch_size_matrix), natom),
                    decompr_idx[begin:end],
                ),
            ),
            shape=(batch_size_matrix, NNN27 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_sum_cplmt += dot_product_sparse(
            c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
        )

    proj_sum_cplmt /= n_lp * natom
    return proj_sum_cplmt


def compressed_projector_sum_rules_from_compact_compr_mat(
    trans_perms, n_a_compress_mat: csr_array, use_mkl: bool = False
) -> csr_array:
    """Return projection matrix for sum rule.

    This is compressed by C_compr = C_trans @ n_a_compress_mat.
    """
    proj_cplmt = compressed_complement_projector_sum_rules_from_compact_compr_mat(
        trans_perms, n_a_compress_mat, use_mkl=use_mkl
    )
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def get_combinations(n, r):
    """Return numpy array of combinations.

    combinations = np.array(
       list(itertools.combinations(range(n), r)), dtype=int
    )
    """
    combs = np.ones((r, n - r + 1), dtype=int)
    combs[0] = np.arange(n - r + 1)
    for j in range(1, r):
        reps = (n - r + j) - combs[j - 1]
        combs = np.repeat(combs, reps, axis=1)
        ind = np.add.accumulate(reps)
        combs[j, ind[:-1]] = 1 - reps[1:]
        combs[j, 0] = j
        combs[j] = np.add.accumulate(combs[j])
    return combs.T


def N3N3N3_to_NNNand333(combinations_perm: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vecNNN, vec333 = np.divmod(combinations_perm, 3)
    vecNNN = vecNNN @ np.array([N**2, N, 1])
    vec333 = vec333 @ np.array([9, 3, 1])
    return vecNNN, vec333


def projector_permutation_lat_trans_O3(
    trans_perms: np.ndarray,
    atomic_decompr_idx: np.ndarray = None,
    fc_cutoff: FCCutoffO3 = None,
    n_batch=None,
    use_mkl=False,
):
    """Calculate a projector for permutation rules compressed by C_trans.

    This is calculated without allocating C_trans and C_perm.
    Batch calculations are used to reduce memory allocation.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.
    fc_cutoff : FCCutoffO3

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    if n_batch is None:
        n_batch = 1 if natom <= 128 else int(round((natom / 128) ** 2))

    # (1) for FC3 with single index ia
    if fc_cutoff is None:
        combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    else:
        combinations = fc_cutoff.combinations1()

    n_perm1 = combinations.shape[0]
    combinations, combinations333 = N3N3N3_to_NNNand333(combinations, natom)

    c_pt = csr_array(
        (
            np.full(n_perm1, 1.0 / np.sqrt(n_lp)),
            (
                np.arange(n_perm1),
                atomic_decompr_idx[combinations] * 27 + combinations333,
            ),
        ),
        shape=(n_perm1, NNN27 // n_lp),
        dtype="double",
    )
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (2) for FC3 with two distinguished indices (ia,ia,jb)
    if fc_cutoff is None:
        combinations = get_combinations(3 * natom, 2)
    else:
        combinations = fc_cutoff.combinations2()

    n_perm2 = combinations.shape[0] * 2
    perms = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    combinations = combinations[:, perms].reshape((-1, 3))
    combinations, combinations333 = N3N3N3_to_NNNand333(combinations, natom)

    c_pt = csr_array(
        (
            np.full(n_perm2 * 3, 1 / np.sqrt(3 * n_lp)),
            (
                np.repeat(range(n_perm2), 3),
                atomic_decompr_idx[combinations] * 27 + combinations333,
            ),
        ),
        shape=(n_perm2, NNN27 // n_lp),
        dtype="double",
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # (3) for FC3 with three distinguished indices (ia,jb,kc)
    """Bottleneck part for memory reduction in constructing a basis set.
    Moreover, combinations can be divided using fc_cut.combiations3(i).
    """
    if fc_cutoff is None:
        combinations = get_combinations(3 * natom, 3)
    else:
        combinations = fc_cutoff.combinations3_all()

    n_perm3 = combinations.shape[0]
    perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    for begin, end in zip(*get_batch_slice(n_perm3, n_perm3 // n_batch)):
        print("Proj (perm.T @ trans):", str(end) + "/" + str(n_perm3))
        batch_size = end - begin
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 3))
        combinations_perm, combinations333 = N3N3N3_to_NNNand333(
            combinations_perm, natom
        )

        c_pt = csr_array(
            (
                np.full(batch_size * 6, 1 / np.sqrt(6 * n_lp)),
                (
                    np.repeat(np.arange(batch_size), 6),
                    atomic_decompr_idx[combinations_perm] * 27 + combinations333,
                ),
            ),
            shape=(batch_size, NNN27 // n_lp),
            dtype="double",
        )
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    return proj_pt
