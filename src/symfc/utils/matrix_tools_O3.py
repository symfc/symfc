"""Matrix utility functions for 3rd order force constants."""

from typing import Optional, Union

import numpy as np
import scipy
from scipy.sparse import csr_array, vstack

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.matrix_tools import permutation_dot_lat_trans
from symfc.utils.permutation_tools import get_combinations
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O3 import get_atomic_lat_trans_decompr_indices_O3

try:
    from symfc.utils.eig_tools import dot_product_sparse
except ImportError:
    pass


def _N3N3N3_to_NNNand333(combs: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
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
    permutations: Union[np.ndarray, list],
    atomic_decompr_idx: np.ndarray,
    trans_perms: np.ndarray,
    included_indices: Optional[np.ndarray] = None,
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
                proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
            else:
                proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
            c_pt = None

    if c_pt is not None:
        if proj_pt is None:
            proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
        else:
            proj_pt += dot_product_sparse(c_pt.T, c_pt, use_mkl=use_mkl)  # type: ignore
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
            included_indices,  # type: ignore
            use_mkl=use_mkl,
        )
    return proj_pt


def optimize_batch_size_sum_rules_O3(natom: int, n_batch: Optional[int] = None):
    """Calculate batch size for constructing projector for sum rules."""
    if n_batch is None:
        if natom < 256:
            n_batch = natom // min(natom, 16)
        else:
            n_batch = natom // 4

    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom**2 * (natom // n_batch)
    return batch_size


def compressed_projector_sum_rules_O3(
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
    1. C_sum^(c).T = [I, I, I, ...] of size (27N^2, 27N^3).
       I denotes the unit matrix of size (27N^2, 27N^2).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC3(i, j, k, a, b, c) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of (i, j, k, a, b, c)
       to (a, b, c, j, k, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, c, j, k, i). Using this permutation, the translational sum rules are
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
       =   [T_1, T_2, ..., T_NN333]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_NN333].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aNN333) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NNN = natom**3
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]  # type: ignore
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NN)).T.reshape(-1) * 27

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    nonzero = np.zeros((natom, natom, natom), dtype=bool)
    nonzero[indep_atoms, :, :] = True
    nonzero = nonzero.reshape(-1)
    if fc_cutoff is not None:
        nonzero_c = fc_cutoff.nonzero_atomic_indices_fc3()
        nonzero_c = nonzero_c.reshape((natom, NN)).T.reshape(-1)
        nonzero = nonzero & nonzero_c

    batch_size = optimize_batch_size_sum_rules_O3(natom, n_batch=n_batch)
    abc = np.arange(27)
    for begin, end in zip(*get_batch_slice(NNN, batch_size)):
        size = end - begin
        size_vector = size * 27
        size_row = size_vector // natom

        nonzero_b = nonzero[begin:end]
        size_data = np.count_nonzero(nonzero_b) * 27
        if size_data == 0:
            continue

        if verbose:
            print("Complementary P (Sum rule):", str(end) + "/" + str(NNN), flush=True)
        decompr_idx_b = decompr_idx[begin:end][nonzero_b]
        c_sum_cplmt = csr_array(
            (
                np.ones(size_data, dtype="double"),
                (
                    np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 27)],
                    (abc[:, None] + decompr_idx_b[None, :]).reshape(-1),
                ),
            ),
            shape=(size_row, NNN27 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= natom
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
    1. C_sum^(c).T = [I, I, I, ...] of size (27N^2, 27N^3).
       I denotes the unit matrix of size (27N^2, 27N^2).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC3(i, j, k, a, b, c) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of (i, j, k, a, b, c)
       to (a, b, c, j, k, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, c, j, k, i). Using this permutation, the translational sum rules are
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
       =   [T_1, T_2, ..., T_NN333]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_NN333].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aNN333) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NNN27 = natom**3 * 27
    NNN = natom**3
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]  # type: ignore
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = get_atomic_lat_trans_decompr_indices_O3(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, NN)).T.reshape(-1) * 27
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc3()
        nonzero = nonzero.reshape((natom, NN)).T.reshape(-1)

    batch_size = optimize_batch_size_sum_rules_O3(natom, n_batch=n_batch)
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
                        (abc[:, None] + decompr_idx[begin:end][None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            decompr_idx_b = decompr_idx[begin:end][nonzero_b]
            size_data = np.count_nonzero(nonzero_b) * 27
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 27)],
                        (abc[:, None] + decompr_idx_b[None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NNN27 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
