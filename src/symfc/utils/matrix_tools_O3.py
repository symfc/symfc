"""Matrix utility functions for 3rd order force constants."""

from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array, vstack

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import get_combinations, permutation_dot_lat_trans
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def _N3N3N3_to_NNNand333(combs: np.ndarray, N: int) -> np.ndarray:
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


def _construct_projector_permutation_lat_trans_from_combinations(
    combinations: np.ndarray,
    permutations: np.ndarray,
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    included_indices: np.ndarray = None,
    n_perms_group: int = 1,
    use_mkl: bool = False,
    n_batch: int = 1,
    verbose: bool = False,
) -> csr_array:
    """Construct projector of permutation and lattice translation."""
    n_lp, natom = trans_perms.shape
    n_comb = combinations.shape[0]
    n_perms = len(permutations)

    c_pt, proj_pt = None, None
    for begin, end in zip(*get_batch_slice(n_comb, n_comb // n_batch)):
        if verbose:
            print("Proj (perm.T @ trans):", str(end) + "/" + str(n_comb), flush=True)
        combs_perm = combinations[begin:end][:, permutations].reshape((-1, 3))
        combs_perm, combs333 = _N3N3N3_to_NNNand333(combs_perm, natom)
        if included_indices is not None:
            included_indices[combs_perm * 27 + combs333] = True

        c_pt_batch = permutation_dot_lat_trans(
            combs_perm,
            combs333,
            atomic_decompr_idx,
            n_perms=n_perms,
            n_perms_group=n_perms_group,
            n_lp=n_lp,
            order=3,
            natom=natom,
        )
        c_pt = c_pt_batch if c_pt is None else vstack([c_pt, c_pt_batch])
        if len(c_pt.data) > 2147483647 // 4:
            if verbose:
                print("Executed: proj_pt += c_pt.T @ c_pt", flush=True)
            if proj_pt is None:
                proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
            else:
                proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
            c_pt = None

    if c_pt is not None:
        if proj_pt is None:
            proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
        else:
            proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)
    return proj_pt


def _projector_permutation_lat_trans_unique_index1(
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    included_indices: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """FC3 with single distinct index (ia, ia, ia)."""
    n_lp, natom = trans_perms.shape
    combinations = np.array([[i, i, i] for i in range(3 * natom)], dtype=int)
    perms = [[0, 0, 0]]
    proj_pt = _construct_projector_permutation_lat_trans_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        included_indices=included_indices,
        n_perms_group=1,
        use_mkl=use_mkl,
        n_batch=1,
        verbose=verbose,
    )
    return proj_pt


def _projector_permutation_lat_trans_unique_index2(
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    included_indices: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    use_mkl: bool = False,
    verbose: bool = False,
) -> csr_array:
    """FC3 with two distinct indices (ia,ia,jb)."""
    n_lp, natom = trans_perms.shape
    combinations = get_combinations(natom, order=2, fc_cutoff=fc_cutoff)
    perms = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    proj_pt = _construct_projector_permutation_lat_trans_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        included_indices=included_indices,
        n_perms_group=2,
        use_mkl=use_mkl,
        n_batch=1,
        verbose=verbose,
    )
    return proj_pt


def _projector_permutation_lat_trans_unique_index3(
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    included_indices: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    use_mkl: bool = False,
    n_batch: int = 1,
    verbose: bool = False,
) -> csr_array:
    """FC3 with three distinct indices (ia,jb,kc)."""
    if fc_cutoff is not None or included_indices is None:
        complete = True
    else:
        complete = False
    if verbose:
        print("Construct complete projector:", complete)

    n_lp, natom = trans_perms.shape
    combinations = get_combinations(natom, order=3, fc_cutoff=fc_cutoff)

    if not complete:
        indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
        nonzero = np.zeros(combinations.shape[0], dtype=bool)
        atom_indices = combinations[:, 0] // 3
        for i in indep_atoms:
            nonzero[atom_indices == i] = True
        combinations = combinations[nonzero]

    perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    proj_pt = _construct_projector_permutation_lat_trans_from_combinations(
        combinations,
        perms,
        atomic_decompr_idx,
        trans_perms,
        included_indices=included_indices,
        n_perms_group=1,
        use_mkl=use_mkl,
        n_batch=n_batch,
        verbose=verbose,
    )
    return proj_pt


def _projector_not_reduced(
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    included_indices: np.ndarray,
    use_mkl: bool = False,
):
    """Calculate a projector for indices that are not reduced by permutation rules."""
    n_lp, natom = trans_perms.shape
    NNN333 = natom**3 * 27
    combinations_cmplt = np.where(included_indices)[0]
    combinations_cmplt, combinations333 = np.divmod(combinations_cmplt, 27)
    col = atomic_decompr_idx[combinations_cmplt] * 27 + combinations333
    c_cmplt = csr_array(
        (
            np.full(len(col), 1 / np.sqrt(n_lp)),
            (
                np.arange(len(col)),
                col,
            ),
        ),
        shape=(len(col), NNN333 // n_lp),
        dtype="double",
    )
    proj_cmplt = dot_product_sparse(c_cmplt.T, c_cmplt, use_mkl=use_mkl)
    return scipy.sparse.identity(NNN333 // n_lp) - proj_cmplt


def projector_permutation_lat_trans_O3(
    trans_perms: np.ndarray,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: Optional[int] = None,
    use_mkl: bool = False,
    verbose: bool = False,
    complete: bool = False,
) -> csr_array:
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
    NNN27 = natom**3 * 27
    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    if n_batch is None:
        n_batch = 1 if natom <= 128 else int(round((natom / 128) ** 2))

    if fc_cutoff is not None:
        complete = True

    included_indices = None if complete else np.zeros(NNN27, dtype=bool)
    proj_pt = _projector_permutation_lat_trans_unique_index1(
        atomic_decompr_idx,
        trans_perms,
        included_indices=included_indices,
        fc_cutoff=fc_cutoff,
        use_mkl=use_mkl,
    )
    proj_pt += _projector_permutation_lat_trans_unique_index2(
        atomic_decompr_idx,
        trans_perms,
        included_indices=included_indices,
        fc_cutoff=fc_cutoff,
        use_mkl=use_mkl,
    )
    proj_pt += _projector_permutation_lat_trans_unique_index3(
        atomic_decompr_idx,
        trans_perms,
        included_indices=included_indices,
        fc_cutoff=fc_cutoff,
        use_mkl=use_mkl,
        n_batch=n_batch,
        verbose=verbose,
    )
    if not complete:
        proj_pt += _projector_not_reduced(
            atomic_decompr_idx,
            trans_perms,
            included_indices,
            use_mkl=use_mkl,
        )
    return proj_pt


def compressed_projector_sum_rules_O3(
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
            n_batch = natom // min(natom, 64)
        else:
            n_batch = natom // 4

    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom**2 * (natom // n_batch)

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    """Bottleneck part when using cutoff distance"""
    decompr_idx = atomic_decompr_idx.reshape((natom, NN)).T.reshape(-1) * 27

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    nonzero = np.zeros((natom, natom, natom), dtype=bool)
    nonzero[indep_atoms, :, :] = True
    nonzero = nonzero.reshape(-1)
    if fc_cutoff is not None:
        nonzero_cutoff = fc_cutoff.nonzero_atomic_indices_fc3()
        nonzero_cutoff = nonzero_cutoff.reshape((natom, NN)).T.reshape(-1)
        nonzero = nonzero & nonzero_cutoff

    abc = np.arange(27)
    for begin, end in zip(*get_batch_slice(NNN, batch_size)):
        size = end - begin
        size_vector = size * 27
        size_row = size_vector // natom

        nonzero_b = nonzero[begin:end]
        size_data = np.count_nonzero(nonzero_b) * 27
        if verbose:
            print(
                "Complementary P (Sum rule):",
                str(end) + "/" + str(NNN) + ",",
                "Non-zero:",
                size_data,
                flush=True,
            )
        if size_data > 0:
            decompr_idx_b = decompr_idx[begin:end]
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 27)],
                        (decompr_idx_b[nonzero_b][None, :] + abc[:, None]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )
            c_sum_cplmt = dot_product_sparse(
                c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl
            )
            proj_cplmt += dot_product_sparse(
                c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl
            )

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt


def compressed_projector_sum_rules_O3_stable(
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
