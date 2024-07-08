"""Matrix utility functions for 3rd order force constants."""

import numpy as np
import scipy
from scipy.sparse import csr_array, vstack

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.eig_tools import dot_product_sparse
from symfc.utils.matrix_tools import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3


def N3N3N3_to_NNNand333(combs: np.ndarray, N: int) -> np.ndarray:
    """Transform index order."""
    vecNNN, vec333 = np.divmod(combs[:, 0], 3)
    vecNNN *= N**2
    vec333 *= 9
    div, mod = np.divmod(combs[:, 1], 3)
    vecNNN += div * N
    vec333 += mod * 3
    div, mod = np.divmod(combs[:, 2], 3)
    vecNNN += div
    vec333 += mod
    return vecNNN, vec333


def projector_permutation_lat_trans_O3(
    trans_perms: np.ndarray,
    atomic_decompr_idx: np.ndarray = None,
    fc_cutoff: FCCutoff = None,
    n_batch: int = None,
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
    fc_cutoff : FCCutoff

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
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
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

    c_pt = None
    for begin, end in zip(*get_batch_slice(n_perm3, n_perm3 // n_batch)):
        if verbose:
            print("Proj (perm.T @ trans):", str(end) + "/" + str(n_perm3), flush=True)
        batch_size = end - begin
        combinations_perm = combinations[begin:end][:, perms].reshape((-1, 3))
        combinations_perm, combinations333 = N3N3N3_to_NNNand333(
            combinations_perm, natom
        )

        c_pt_batch = csr_array(
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

        c_pt = c_pt_batch if c_pt is None else vstack([c_pt, c_pt_batch])

    proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
    return proj_pt


def compressed_projector_sum_rules_O3(
    trans_perms,
    n_a_compress_mat: csr_array,
    fc_cutoff: FCCutoff = None,
    atomic_decompr_idx: np.ndarray = None,
    use_mkl: bool = False,
    n_batch: int = None,
    verbose: bool = False,
) -> csr_array:
    """Return projection matrix for sum rule.

    Calculate a complementary projector for sum rules.
    This is compressed by C_trans and n_a_compress_mat without
    allocating C_trans.
    Memory efficient version using get_atomic_lat_trans_decompr_indices_O3.

    Return
    ------
    Compressed projector I - P^(c)
    P^(c) = n_a_compress_mat.T @ C_trans.T @ C_sum^(c)
            @ C_sum^(c).T @ C_trans @ n_a_compress_mat
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NNN = natom**3
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if n_batch is None:
        if natom < 256:
            n_batch = natom // min(natom, 16)
        else:
            n_batch = natom // 4

    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom**2 * (natom // n_batch)

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NN)).T.reshape(-1) * 27
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc3()
        nonzero = nonzero.reshape((natom, NN)).T.reshape(-1)

    abc = np.arange(27)
    for begin, end in zip(*get_batch_slice(NNN, batch_size)):
        if verbose:
            print("Complementary P (Sum rule):", str(end) + "/" + str(NNN), flush=True)
        size = end - begin
        size_vector = size * 27
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
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            size_data = np.count_nonzero(nonzero_b) * 27
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 27)],
                        (
                            decompr_idx[begin:end][nonzero_b][None, :] + abc[:, None]
                        ).reshape(-1),
                    ),
                ),
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
