"""Matrix utility functions for 4th order force constants."""

import itertools
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array, vstack

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import get_combinations, permutation_dot_lat_trans
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O4 import get_atomic_lat_trans_decompr_indices_O4

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def N3N3N3N3_to_NNNNand3333(combs: np.ndarray, N: int) -> np.ndarray:
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
            proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
            c_pt = None

    if c_pt is not None:
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)

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
            proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
            c_pt = None

    if c_pt is not None:
        proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
    return proj_pt


def compressed_projector_sum_rules_O4(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: Optional[int] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """Return projection matrix for sum rule.

    Calculate a complementary projector for sum rules.
    This is compressed by C_trans and n_a_compress_mat without
    allocating C_trans.
    Memory efficient version using get_atomic_lat_trans_decompr_indices_O4.

    Return
    ------
    Compressed projector I - P^(c)
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """
    n_lp, natom = trans_perms.shape
    NNNN81 = natom**4 * 81
    NNNN = natom**4
    NNN = natom**3

    proj_size = n_a_compress_mat.shape[1]
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    n_batch = 1 if natom < 12 else natom // 3
    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom**3 * (natom // n_batch)

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O4(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NNN)).T.reshape(-1) * 81
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc4()
        nonzero = nonzero.reshape((natom, NNN)).T.reshape(-1)

    abc = np.arange(81)
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
                        (decompr_idx[begin:end][None, :] + abc[:, None]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNNN81 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            size_data = np.count_nonzero(nonzero_b) * 81
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 81)],
                        (
                            decompr_idx[begin:end][nonzero_b][None, :] + abc[:, None]
                        ).reshape(-1),
                    ),
                ),
                shape=(size_row, NNNN81 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
