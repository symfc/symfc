"""Matrix utility functions for 2nd order force constants."""

from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import permutation_dot_lat_trans
from symfc.utils.permutation_tools import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def N3N3_to_NNand33(combs: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vecNN, vec33 = np.divmod(combs[:, 0], 3)
    vecNN *= N
    vec33 *= 3
    div, mod = np.divmod(combs[:, 1], 3)
    vecNN += div
    vec33 += mod
    return vecNN, vec33


def projector_permutation_lat_trans_O2(
    trans_perms: np.ndarray,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    use_mkl: bool = False,
    verbose: bool = False,
):
    """Calculate a projector for permutation rules compressed by C_trans.

    This is calculated without allocating C_trans and C_perm.

    Parameters
    ----------
    trans_perms : ndarray
        Permutation of atomic indices by lattice translational symmetry.
        dtype='intc'.
        shape=(n_l, N), where n_l and N are the numbers of lattce points and
        atoms in supercell.
    atomic_decompr_idx: ndarray
        Indices of atomic lattice translation compression matrix.
        Default is None.
    fc_cutoff : FCCutoff class object. Default is None.

    Return
    ------
    Compressed projector for permutation.
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    # FC2 with single distinct index (ia, ia)
    combinations = np.array([[i, i] for i in range(3 * natom)], dtype=int)
    combinations, combinations33 = N3N3_to_NNand33(combinations, natom)
    c_pt = permutation_dot_lat_trans(
        combinations,
        combinations33,
        atomic_decompr_idx,
        n_perms=1,
        n_perms_group=1,
        n_lp=n_lp,
        order=2,
        natom=natom,
    )
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    # FC2 with two distinct indices (ia, jb)
    combinations = get_combinations(natom, order=2, fc_cutoff=fc_cutoff)
    perms = [[0, 1], [1, 0]]
    combinations = combinations[:, perms].reshape((-1, 2))
    combinations, combinations33 = N3N3_to_NNand33(combinations, natom)
    c_pt = permutation_dot_lat_trans(
        combinations,
        combinations33,
        atomic_decompr_idx,
        n_perms=len(perms),
        n_perms_group=1,
        n_lp=n_lp,
        order=2,
        natom=natom,
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
    return proj_pt


def optimize_batch_size_sum_rules_O2(natom: int, n_batch: Optional[int] = None):
    """Calculate batch size for constructing projector for sum rules."""
    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom * (natom // n_batch)
    return batch_size


def compressed_projector_sum_rules_O2(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: int = 1,
    use_mkl: bool = False,
) -> csr_array:
    r"""Return projection matrix for translational sum rule.

    Calculate a compressed projector for translational sum rules
    efficiently using independent atom with respect to lattice translations.
    This compression is achieved using C_trans and n_a_compress_mat,
    without the need to allocate C_trans. The implementation utilizes
    get_atomic_lat_trans_decompr_indices_O3 to ensure efficient memory usage.

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
    1. C_sum^(c).T = [I, I, I, ...] of size (27N, 27N^2).
       I denotes the unit matrix of size (27N, 27N).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC2(i, j, a, b) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of (i, j, a, b)
       to (a, b, j, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, j, i). Using this permutation, the translational sum rules are
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
       =   [T_1, T_2, ..., T_N33]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_N33].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aN33) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NN9 = natom**2 * 9
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, natom)).T.reshape(-1) * 9

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    nonzero = np.zeros((natom, natom), dtype=bool)
    nonzero[indep_atoms, :] = True
    nonzero = nonzero.reshape(-1)
    if fc_cutoff is not None:
        nonzero_c = fc_cutoff.nonzero_atomic_indices_fc2()
        nonzero_c = nonzero_c.reshape((natom, natom)).T.reshape(-1)
        nonzero = nonzero & nonzero_c

    batch_size = optimize_batch_size_sum_rules_O2(natom, n_batch=n_batch)
    ab = np.arange(9)
    for begin, end in zip(*get_batch_slice(NN, batch_size)):
        size = end - begin
        size_vector = size * 9
        size_row = size_vector // natom

        nonzero_b = nonzero[begin:end]
        size_data = np.count_nonzero(nonzero_b) * 9
        if size_data == 0:
            continue

        decompr_idx_b = decompr_idx[begin:end][nonzero_b]
        c_sum_cplmt = csr_array(
            (
                np.ones(size_data, dtype="double"),
                (
                    np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 9)],
                    (ab[:, None] + decompr_idx_b[None, :]).reshape(-1),
                ),
            ),
            shape=(size_row, NN9 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def compressed_projector_sum_rules_O2_stable(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: int = 1,
    use_mkl: bool = False,
) -> csr_array:
    r"""Return projection matrix for translational sum rule.

    Calculate a compressed projector for translational sum rules.
    This compression is achieved using C_trans and n_a_compress_mat,
    without the need to allocate C_trans. The implementation utilizes
    get_atomic_lat_trans_decompr_indices_O3 to ensure efficient memory usage.

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
    1. C_sum^(c).T = [I, I, I, ...] of size (27N, 27N^2).
       I denotes the unit matrix of size (27N, 27N).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC2(i, j, a, b) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of (i, j, a, b)
       to (a, b, j, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, j, i). Using this permutation, the translational sum rules are
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
       =   [T_1, T_2, ..., T_N33]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_N33].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aN33) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NN9 = natom**2 * 9
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, natom)).T.reshape(-1) * 9
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc2()
        nonzero = nonzero.reshape((natom, natom)).T.reshape(-1)

    batch_size = optimize_batch_size_sum_rules_O2(natom, n_batch=n_batch)
    ab = np.arange(9)
    for begin, end in zip(*get_batch_slice(NN, batch_size)):
        size = end - begin
        size_vector = size * 9
        size_row = size_vector // natom

        if fc_cutoff is None:
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_vector, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom),
                        (ab[:, None] + decompr_idx[begin:end][None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NN9 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            decompr_idx_b = decompr_idx[begin:end][nonzero_b]
            size_data = np.count_nonzero(nonzero_b) * 9
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 9)],
                        (ab[:, None] + decompr_idx_b[None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NN9 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
