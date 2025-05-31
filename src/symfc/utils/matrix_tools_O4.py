"""Matrix utility functions for 4th order force constants."""

import itertools
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array, vstack

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import permutation_dot_lat_trans
from symfc.utils.permutation_tools import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def N3N3N3N3_to_NNNNand3333(combs: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform index order."""
    vecNNNN, vec3333 = np.divmod(combs[:, 0], 3)
    vecNNNN *= N**3
    vec3333 *= 27
    div, mod = np.divmod(combs[:, 1], 3)
    vecNNNN += div * N**2
    vec3333 += mod * 9
    div, mod = np.divmod(combs[:, 2], 3)
    vecNNNN += div * N
    vec3333 += mod * 3
    div, mod = np.divmod(combs[:, 3], 3)
    vecNNNN += div
    vec3333 += mod
    return vecNNNN, vec3333


def projector_permutation_lat_trans_O4(
    trans_perms: np.ndarray,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    use_mkl: bool = False,
    verbose: bool = False,
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
    fc_cutoff : FCCutoff class object. Default is None.

    Return
    ------
    Compressed projector for permutation
    P_pt = C_trans.T @ C_perm @ C_perm.T @ C_trans
    """
    n_lp, natom = trans_perms.shape
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    """FC4 with single distinct index (ia, ia, ia, ia)"""
    combinations = np.array([[i, i, i, i] for i in range(3 * natom)], dtype=int)
    combinations, combinations3333 = N3N3N3N3_to_NNNNand3333(combinations, natom)
    c_pt = permutation_dot_lat_trans(
        combinations,
        combinations3333,
        atomic_decompr_idx,
        n_perms=1,
        n_perms_group=1,
        n_lp=n_lp,
        order=4,
        natom=natom,
    )
    proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    """FC4 with two distinct indices (ia,ia,ia,jb)"""
    combinations = get_combinations(natom, order=2, fc_cutoff=fc_cutoff)
    perms = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ]
    combinations = combinations[:, perms].reshape((-1, 4))
    combinations, combinations3333 = N3N3N3N3_to_NNNNand3333(combinations, natom)

    c_pt = permutation_dot_lat_trans(
        combinations,
        combinations3333,
        atomic_decompr_idx,
        n_perms=len(perms),
        n_perms_group=2,
        n_lp=n_lp,
        order=4,
        natom=natom,
    )
    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

    """FC4 with three distinct indices (ia,ia,jb,kc)
    [
        [a, a, b, c],
        [a, a, c, b],
        [a, b, a, c],
        [a, c, a, b],
        [a, b, c, a],
        [a, c, b, a],
        [b, a, a, c],
        [c, a, a, b],
        [b, a, c, a],
        [c, a, b, a],
        [b, c, a, a],
        [c, b, a, a],
    ]
    """
    if verbose:
        print("Find combinations of three FC elements", flush=True)

    combinations = get_combinations(natom, order=3, fc_cutoff=fc_cutoff)
    n_comb3 = combinations.shape[0]
    perms = [
        [0, 0, 1, 2],
        [0, 0, 2, 1],
        [0, 1, 0, 2],
        [0, 2, 0, 1],
        [0, 1, 2, 0],
        [0, 2, 1, 0],
        [1, 0, 0, 2],
        [2, 0, 0, 1],
        [1, 0, 2, 0],
        [2, 0, 1, 0],
        [1, 2, 0, 0],
        [2, 1, 0, 0],
        [1, 1, 0, 2],
        [1, 1, 2, 0],
        [1, 0, 1, 2],
        [1, 2, 1, 0],
        [1, 0, 2, 1],
        [1, 2, 0, 1],
        [0, 1, 1, 2],
        [2, 1, 1, 0],
        [0, 1, 2, 1],
        [2, 1, 0, 1],
        [0, 2, 1, 1],
        [2, 0, 1, 1],
        [2, 2, 1, 0],
        [2, 2, 0, 1],
        [2, 1, 2, 0],
        [2, 0, 2, 1],
        [2, 1, 0, 2],
        [2, 0, 1, 2],
        [1, 2, 2, 0],
        [0, 2, 2, 1],
        [1, 2, 0, 2],
        [0, 2, 1, 2],
        [1, 0, 2, 2],
        [0, 1, 2, 2],
    ]

    n_batch3 = (n_comb3 // 100000) + 1
    c_pt = None
    for begin, end in zip(*get_batch_slice(n_comb3, n_comb3 // n_batch3)):
        if verbose:
            print(
                "Proj (perm.T @ trans, 3):", str(end) + "/" + str(n_comb3), flush=True
            )
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 4))
        combinations_perm, combinations3333 = N3N3N3N3_to_NNNNand3333(
            combinations_perm, natom
        )
        c_pt_batch = permutation_dot_lat_trans(
            combinations_perm,
            combinations3333,
            atomic_decompr_idx,
            n_perms=len(perms),
            n_perms_group=3,
            n_lp=n_lp,
            order=4,
            natom=natom,
        )
        c_pt = c_pt_batch if c_pt is None else vstack([c_pt, c_pt_batch])
        if len(c_pt.data) > 2147483647 / 4:
            if verbose:
                print("Executed: proj_pt += c_pt.T @ c_pt", flush=True)
            proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
            c_pt = None

    if c_pt is not None:
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore

    """FC4 with four distinct indices (ia,jb,kc,ld)"""
    if verbose:
        print("Find combinations of four FC elements", flush=True)

    combinations = get_combinations(natom, order=4, fc_cutoff=fc_cutoff)
    n_comb4 = combinations.shape[0]
    perms = np.array(list(itertools.permutations(range(4))))

    n_batch4 = (n_comb4 // 100000) + 1
    c_pt = None
    for begin, end in zip(*get_batch_slice(n_comb4, n_comb4 // n_batch4)):
        if verbose:
            print(
                "Proj (perm.T @ trans, 4):", str(end) + "/" + str(n_comb4), flush=True
            )
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 4))
        combinations_perm, combinations3333 = N3N3N3N3_to_NNNNand3333(
            combinations_perm, natom
        )
        c_pt_batch = permutation_dot_lat_trans(
            combinations_perm,
            combinations3333,
            atomic_decompr_idx,
            n_perms=len(perms),
            n_perms_group=1,
            n_lp=n_lp,
            order=4,
            natom=natom,
        )
        c_pt = c_pt_batch if c_pt is None else vstack([c_pt, c_pt_batch])

        if len(c_pt.data) > 2147483647 / 4:
            if verbose:
                print("Executed: proj_pt += c_pt.T @ c_pt", flush=True)
            proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
            c_pt = None

    if c_pt is not None:
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
    return proj_pt


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
