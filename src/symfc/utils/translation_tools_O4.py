"""Functions for introducing translational invariance in 4th order force constants."""

from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass


def optimize_batch_size_sum_rules_O4(natom: int, n_batch: Optional[int] = None):
    """Calculate batch size for constructing projector for sum rules."""
    if n_batch is None:
        if natom < 32:
            n_batch = natom // min(natom, 8)
        else:
            n_batch = natom // 4

    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom**3 * (natom // n_batch)
    return batch_size


def compressed_projector_sum_rules_O4(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: Optional[int] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    r"""Return projection matrix for translational sum rule.

    Calculate a compressed projector for translational sum rules
    efficiently using independent atom with respect to lattice translations.
    This compression is achieved using C_trans and n_a_compress_mat,
    without the need to allocate C_trans. The implementation utilizes
    get_atomic_lat_trans_decompr_indices_O4 to ensure efficient memory usage.

    Return
    ------
    Compressed projector I - P^(c).
    I - P^(c)
    = n_a_compress_mat.T @ C_trans.T
      @ [I - C_sum^(c) @ C_sum^(c).T] @ C_trans @ n_a_compress_mat
    = I - [n_a_compress_mat.T @ C_trans.T @ C_sum^(c)]
          @ [C_sum^(c).T @ C_trans @ n_a_compress_mat]

    Algorithm
    ---------
    1. C_sum^(c).T = [I, I, I, ...] of size (27N^3, 27N^4).
       I denotes the unit matrix of size (27N^3, 27N^3).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC4(i, j, k, l, a, b, c, d) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of
       (i, j, k, l, a, b, c, d) to (a, b, c, d, j, k, l, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, c, d, j, k, l, i). Using this permutation, the translational sum rules are
       represented as
       C_sum^(c).T @ S.T = [
           [1_N.T, 0_N.T, 0_N.T, ...]
           [0_N.T, 1_N.T, 0_N.T, ...]
           [0_N.T, 0_N.T, 1_N.T, ...]
           ...
       ],
       where 1_N and 0_N are column vectors of size N with all elements
       equal to one and zero, respectively.
       (Example) C_sum^(c).T @ S.T = [
                    [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 ...]
                    [0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 ...]
                    [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 ...]
                    ...
                 ]
        In this function, the permutation is achieved by using matrix reshapes.

    3. Set C_trans.T @ C_sum^(c) @ C_sum^(c).T @ C_trans
       = [(C_trans.T @ S.T) @ (S @ C_sum^(c))] @ [(C_sum^(c).T @ S.T) @ (S @ C_trans)]
       =   [T_1, T_2, ..., T_NNN3333]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_NNN3333].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aNNN333) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NNNN81 = natom**4 * 81
    NNNN = natom**4
    NNN = natom**3

    proj_size = n_a_compress_mat.shape[1]  # type: ignore
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NNN)).T.reshape(-1) * 81

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    nonzero = np.zeros((natom, natom, natom, natom), dtype=bool)
    nonzero[indep_atoms, :, :, :] = True
    nonzero = nonzero.reshape(-1)
    if fc_cutoff is not None:
        nonzero_c = fc_cutoff.nonzero_atomic_indices_fc4()
        nonzero_c = nonzero_c.reshape((natom, NNN)).T.reshape(-1)
        nonzero = nonzero & nonzero_c

    batch_size = optimize_batch_size_sum_rules_O4(natom, n_batch=n_batch)
    abcd = np.arange(81)
    for begin, end in zip(*get_batch_slice(NNNN, batch_size)):
        size = end - begin
        size_vector = size * 81
        size_row = size_vector // natom

        nonzero_b = nonzero[begin:end]
        size_data = np.count_nonzero(nonzero_b) * 81
        if size_data == 0:
            continue

        if verbose:
            print("Complementary P (Sum rule):", str(end) + "/" + str(NNN), flush=True)
        decompr_idx_b = decompr_idx[begin:end][nonzero_b]
        c_sum_cplmt = csr_array(
            (
                np.ones(size_data, dtype="double"),
                (
                    np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 81)],
                    (abcd[:, None] + decompr_idx_b[None, :]).reshape(-1),
                ),
            ),
            shape=(size_row, NNNN81 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def compressed_projector_sum_rules_O4_stable(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: Optional[int] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    r"""Return projection matrix for translational sum rule.

    Calculate a compressed projector for translational sum rules.
    This compression is achieved using C_trans and n_a_compress_mat,
    without the need to allocate C_trans. The implementation utilizes
    get_atomic_lat_trans_decompr_indices_O4 to ensure efficient memory usage.

    Return
    ------
    Compressed projector I - P^(c).
    I - P^(c)
    = n_a_compress_mat.T @ C_trans.T
      @ [I - C_sum^(c) @ C_sum^(c).T] @ C_trans @ n_a_compress_mat
    = I - [n_a_compress_mat.T @ C_trans.T @ C_sum^(c)]
          @ [C_sum^(c).T @ C_trans @ n_a_compress_mat]

    Algorithm
    ---------
    1. C_sum^(c).T = [I, I, I, ...] of size (27N^3, 27N^4).
       I denotes the unit matrix of size (27N^3, 27N^3).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC4(i, j, k, l, a, b, c, d) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of
       (i, j, k, l, a, b, c, d) to (a, b, c, d, j, k, l, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, c, d, j, k, l, i). Using this permutation, the translational sum rules are
       represented as
       C_sum^(c).T @ S.T = [
           [1_N.T, 0_N.T, 0_N.T, ...]
           [0_N.T, 1_N.T, 0_N.T, ...]
           [0_N.T, 0_N.T, 1_N.T, ...]
           ...
       ],
       where 1_N and 0_N are column vectors of size N with all elements
       equal to one and zero, respectively.
       (Example) C_sum^(c).T @ S.T = [
                    [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 ...]
                    [0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 ...]
                    [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 ...]
                    ...
                 ]
        In this function, the permutation is achieved by using matrix reshapes.

    3. Set C_trans.T @ C_sum^(c) @ C_sum^(c).T @ C_trans
       = [(C_trans.T @ S.T) @ (S @ C_sum^(c))] @ [(C_sum^(c).T @ S.T) @ (S @ C_trans)]
       =   [T_1, T_2, ..., T_NNN3333]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_NNN3333].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aNNN333) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NNNN81 = natom**4 * 81
    NNNN = natom**4
    NNN = natom**3

    proj_size = n_a_compress_mat.shape[1]  # type: ignore
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NNN)).T.reshape(-1) * 81
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc4()
        nonzero = nonzero.reshape((natom, NNN)).T.reshape(-1)

    batch_size = optimize_batch_size_sum_rules_O4(natom, n_batch=n_batch)
    abcd = np.arange(81)
    for begin, end in zip(*get_batch_slice(NNNN, batch_size)):
        if verbose:
            print("Complementary P (Sum rule):", str(end) + "/" + str(NNNN), flush=True)
        size = end - begin
        size_vector = size * 81
        size_row = size_vector // natom

        if fc_cutoff is None:
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_vector, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom),
                        (abcd[:, None] + decompr_idx[begin:end][None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNNN81 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            decompr_idx_b = decompr_idx[begin:end][nonzero_b]
            size_data = np.count_nonzero(nonzero_b) * 81
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 81)],
                        (abcd[:, None] + decompr_idx_b[None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNNN81 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
