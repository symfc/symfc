"""Utility functions for eigenvalue solutions."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
from scipy.linalg import get_lapack_funcs
from scipy.sparse import csr_array, hstack, lil_array

from symfc.utils.solver_funcs import get_batch_slice

try:
    from sparse_dot_mkl import dot_product_mkl
except ImportError:
    pass


def dot_product_sparse(
    A: csr_array,
    B: csr_array,
    use_mkl: bool = False,
    dense: bool = False,
) -> csr_array:
    """Compute dot-product of sparse matrices."""
    if use_mkl:
        return dot_product_mkl(A, B, dense=dense)
    return A @ B


def _compr_projector(p: csr_array) -> csr_array:
    """Compress projection matrix p with many zero rows and columns."""
    _, col_p = p.nonzero()
    col_p = np.unique(col_p)
    size = len(col_p)

    if p.shape[1] > size:
        compr = csr_array(
            (np.ones(size), (col_p, np.arange(size))),
            shape=(p.shape[1], size),
            dtype="int",
        )
        """p = compr.T @ p @ compr"""
        p = p[col_p].T
        p = p[col_p].T
        return p, compr
    return p, None


def _find_projector_blocks(p: csr_array):
    """Find block structures in projection matrix."""
    n_components, labels = scipy.sparse.csgraph.connected_components(p)
    group = defaultdict(list)
    for i, ll in enumerate(labels):
        group[ll].append(i)
    return group


def _recover_eigvecs_from_uniq_eigvecs(
    uniq_eigvecs: dict,
    group: dict,
    size_projector: int,
):
    """Recover all eigenvectors from unique eigenvectors.

    Parameters
    ----------
    uniq_eigvecs: Unique eigenvectors and submatrixblock indices
                  are included in its values.
    group: Row indices comprising submatrix blocks.

    """
    total_length = sum(
        len(labels) * v.shape[0] * v.shape[1]
        for v, labels in uniq_eigvecs.values()
        if v is not None
    )
    row = np.zeros(total_length, dtype=int)
    col = np.zeros(total_length, dtype=int)
    data = np.zeros(total_length, dtype="double")

    current_id, col_id = 0, 0
    for eigvecs, labels in uniq_eigvecs.values():
        if eigvecs is not None:
            n_row, n_col = eigvecs.shape
            num_labels = len(labels)
            end_id = current_id + n_row * n_col * num_labels

            row[current_id:end_id] = np.repeat(
                [i for ll in labels for i in group[ll]], n_col
            )
            col[current_id:end_id] = [
                j
                for seq, _ in enumerate(labels)
                for i in range(n_row)
                for j in range(col_id + seq * n_col, col_id + (seq + 1) * n_col)
            ]
            data[current_id:end_id] = np.tile(eigvecs.flatten(), num_labels)

            col_id += n_col * num_labels
            current_id = end_id

    n_col = col_id
    eigvecs = csr_array((data, (row, col)), shape=(size_projector, n_col))
    return eigvecs


@dataclass
class DataCSR:
    """Dataclass for extracting data in projector."""

    data: np.ndarray
    block_labels: np.ndarray
    block_sizes: np.ndarray
    slice_begin: Optional[np.ndarray] = None
    slice_end: Optional[np.ndarray] = None

    def __post_init__(self):
        """Init method."""
        self.slice_end = np.cumsum(self.block_sizes**2)
        self.slice_begin = np.zeros_like(self.slice_end)
        self.slice_begin[1:] = self.slice_end[:-1]

    def get_data(self, idx: int):
        """Get data in projector for i-th block."""
        s1 = self.slice_begin[idx]
        s2 = self.slice_end[idx]
        return self.data[s1:s2]

    def get_block_label(self, idx: int):
        """Get block label for i-th block."""
        return self.block_labels[idx]

    def get_block_size(self, idx: int):
        """Get block size for i-th block."""
        return self.block_sizes[idx]

    @property
    def n_blocks(self):
        """Return number of blocks in projector."""
        return len(self.block_labels)


def _extract_sparse_projector_data(p: csr_array, group: dict) -> DataCSR:
    """Extract data in projector in csr_format efficiently.

    Parameters
    ----------
    p: Projection matrix in CSR format.
    group: Row indices comprising submatrix blocks.

    """
    # r = np.array([i for ids in group.values() for i in ids for j in ids])
    group_ravel = [i for ids in group.values() for i in ids]
    lengths = [len(ids) for ids in group.values() for i in ids]
    r = np.repeat(group_ravel, lengths)
    c = np.array([j for ids in group.values() for _ in ids for j in ids])
    sizes = np.array([len(ids) for ids in group.values()], dtype=int)

    p_data = DataCSR(
        data=np.ravel(p[r, c]),
        block_labels=list(group.keys()),
        block_sizes=sizes,
    )
    return p_data


def eigh_projector(
    p: np.ndarray,
    return_complement: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Solve eigenvalue problem using numpy and eliminate eigenvectors with e < 1.0."""
    rank = int(round(np.trace(p)))
    if rank == 0:
        if return_complement:
            return None, None
        return None

    if rank < 32768:
        eigvals, eigvecs = np.linalg.eigh(p)
    else:
        if verbose:
            print("Eigsh_solver: lapack dsyevr is used.", flush=True)
        (syevr,) = get_lapack_funcs(("syevr",), ilp64=False)
        eigvals, eigvecs, _, _, _ = syevr(p, compute_v=True)

    tol = 1e-8
    if np.count_nonzero((eigvals > 1.0 + tol) | (eigvals < -tol)):
        raise ValueError("Eigenvalue error: e > 1 or e < 0.")

    nonzero = np.isclose(eigvals, 1.0)
    if return_complement:
        compr_bool = np.logical_not(nonzero)
        return (
            eigvecs[:, nonzero],
            (eigvals[compr_bool], eigvecs[:, compr_bool]),
        )
    return eigvecs[:, nonzero]


def eigsh_projector(p: csr_array, verbose: bool = True) -> csr_array:
    """Solve eigenvalue problem for matrix p.

    Return sparse matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved. When p = diag(A,B), Av =
    v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w] are solutions.

    This function avoids solving numpy.eigh for duplicate block matrices. This
    function is efficient for matrix p composed of many duplicate block
    matrices.

    """
    p, compr_p = _compr_projector(p)
    group = _find_projector_blocks(p)
    if verbose:
        rank = int(round(sum(p.diagonal())))
        print("Rank of projector:", rank, flush=True)
        print("Number of blocks in projector:", len(group), flush=True)

    p_data = _extract_sparse_projector_data(p, group)
    uniq_eigvecs = dict()
    for i in range(p_data.n_blocks):
        p_block = p_data.get_data(i)
        block_label = p_data.get_block_label(i)
        block_size = p_data.get_block_size(i)
        if block_size > 1:
            key = tuple(p_block)
            try:
                uniq_eigvecs[key][1].append(block_label)
            except KeyError:
                p_np = p_block.reshape((block_size, block_size))
                eigvecs = eigh_projector(p_np, verbose=verbose)
                uniq_eigvecs[key] = [eigvecs, [block_label]]
        else:
            if not np.isclose(p_block[0], 0.0):
                if "one" in uniq_eigvecs:
                    uniq_eigvecs["one"][1].append(block_label)
                else:
                    uniq_eigvecs["one"] = [np.array([[1.0]]), [block_label]]

    c_p = _recover_eigvecs_from_uniq_eigvecs(uniq_eigvecs, group, p.shape[0])
    if compr_p is not None:
        return compr_p @ c_p
    return c_p


def _block_eigh_projector(p_block: np.ndarray, verbose: bool = False):
    """Solve eigenvalue problem using block divisions."""
    eigvecs_block = np.zeros(p_block.shape, dtype="double")
    cmplt = np.zeros((p_block.shape[0], p_block.shape[0]), dtype="double")
    # TODO: memory allocation of cmlpt should be more efficient
    # cmplt = np.zeros((p_block.shape[0], p_block.shape[0] // 2), dtype="double")

    p_size = p_block.shape[0]
    target_size = min(max(p_size // 10, 1000), 3000)

    col_id, col_id_cmplt = 0, 0
    for begin, end in zip(*get_batch_slice(p_size, target_size)):
        if verbose:
            print("Block:", end, "/", p_size, flush=True)
        p_small = p_block[begin:end, begin:end]
        rank = int(round(np.trace(p_small)))
        if rank > 0:
            eigvecs, (cmplt_eigvals, cmplt_small) = eigh_projector(
                p_small,
                return_complement=True,
                verbose=verbose,
            )
            col_end = col_id + eigvecs.shape[1]
            col_end_cmplt = col_id_cmplt + cmplt_small.shape[1]
            eigvecs_block[begin:end, col_id:col_end] = eigvecs
            cmplt[begin:end, col_id_cmplt:col_end_cmplt] = cmplt_small
            p_block[begin:end, begin:end] -= eigvecs @ eigvecs.T
            col_id = col_end
            col_id_cmplt = col_end_cmplt
            if verbose:
                print(eigvecs.shape[1], "eigenvectors are found.", flush=True)

    rank = int(round(np.trace(p_block)))
    if rank > 0:
        if verbose:
            print("Solving complementary projector.", flush=True)
        cmplt = cmplt[:, :col_end_cmplt]
        p_block_rem = cmplt.T @ p_block @ cmplt
        eigvecs = eigh_projector(p_block_rem, verbose=verbose)
        if verbose:
            print(eigvecs.shape[1], "eigenvectors are found.", flush=True)
        if eigvecs.shape[1] > 0:
            col_end = col_id + eigvecs.shape[1]
            eigvecs_block[:, col_id:col_end] = cmplt @ eigvecs
            col_id = col_end

    if col_id == 0:
        return None
    return eigvecs_block[:, :col_id]


def eigsh_projector_sumrule(
    p: csr_array,
    size_threshold: int = 1000,
    verbose: bool = True,
) -> np.ndarray:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.
    """
    if p.shape[0] > size_threshold:
        return eigsh_projector_sumrule_large(p, verbose=verbose)
    return eigsh_projector_sumrule_stable(p, verbose=verbose)


def eigsh_projector_sumrule_stable(p: csr_array, verbose: bool = True) -> np.ndarray:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    This function solves numpy.eigh for all block matrices.
    This function is efficient for matrix p composed of nonequivalent
    block matrices.

    """
    group = _find_projector_blocks(p)
    if verbose:
        print("Use standard normal eigsh solver.", flush=True)
        print("Number of blocks in projector (Sum rule):", len(group), flush=True)

    eigvecs_full = np.zeros(p.shape, dtype="double")
    col_id = 0
    for i, ids in enumerate(group.values()):
        if verbose:
            print("Eigsh_solver_block:", i + 1, "/", len(group), flush=True)
            print(" - Block_size:", len(ids), flush=True)
        p_block = p[np.ix_(ids, ids)].toarray()
        rank = int(round(np.trace(p_block)))
        if rank > 0:
            eigvecs = eigh_projector(p_block, verbose=verbose)
            col_end = col_id + eigvecs.shape[1]
            eigvecs_full[ids, col_id:col_end] = eigvecs
            col_id = col_end
    return eigvecs_full[:, :col_id]


def eigsh_projector_sumrule_large(p: csr_array, verbose: bool = True) -> np.ndarray:
    """Solve eigenvalue problem for matrix p.

    Return dense matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved.
    When p = diag(A,B), Av = v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w]
    are solutions.

    This function solves numpy.eigh for all block matrices.
    This function is efficient for matrix p composed of nonequivalent
    block matrices.

    """
    group = _find_projector_blocks(p)
    if verbose:
        print("Use eigsh solver for large matrices.", flush=True)
        print("Number of blocks in projector (Sum rule):", len(group), flush=True)

    eigvecs_full = np.zeros(p.shape, dtype="double")
    col_id = 0
    for i, ids in enumerate(group.values()):
        if verbose and len(ids) > 2:
            print("Eigsh_solver_block:", i + 1, "/", len(group), flush=True)
            print(" - Block_size:", len(ids), flush=True)
        ids = np.array(ids)
        p_block = p[np.ix_(ids, ids)].toarray()
        rank = int(round(np.trace(p_block)))
        if rank > 0:
            eigvecs = _block_eigh_projector(p_block, verbose=verbose)
            if eigvecs is not None:
                col_end = col_id + eigvecs.shape[1]
                eigvecs_full[ids, col_id:col_end] = eigvecs
                col_id = col_end
    return eigvecs_full[:, :col_id]


def _find_smaller_block(p1: np.ndarray, target_size: int = 3000, random: bool = True):
    """Find a reasonable block in matrix p1."""
    if random:
        # algorithm 2 (random choice)
        total_size = p1.shape[0]
        ids = np.random.choice(
            range(total_size),
            min(target_size, total_size),
            replace=False,
        )
        bools = np.zeros(p1.shape[0], dtype=bool)
        bools[ids] = True
        return bools

    # algorithm 1
    n_data = np.count_nonzero(np.abs(p1) > 1e-15, axis=1)
    rep_id = np.abs(n_data - target_size).argmin()
    return np.abs(p1[rep_id]) > 1e-15


def _iterative_eigsh_projector(
    p_block: np.ndarray,
    max_iter: int = 50,
    size_terminate: int = 2000,
    use_mkl: bool = False,
    verbose: bool = False,
    use_sparse: bool = True,
):
    if use_sparse:
        return _iterative_eigsh_projector_use_sparse(
            p_block=p_block,
            max_iter=max_iter,
            size_terminate=size_terminate,
            use_mkl=use_mkl,
            verbose=verbose,
        )
    return _iterative_eigsh_projector_use_dense(
        p_block=p_block,
        max_iter=max_iter,
        size_terminate=size_terminate,
        verbose=verbose,
    )


def _iterative_eigsh_projector_use_dense(
    p_block: np.ndarray,
    max_iter: int = 50,
    size_terminate: int = 2000,
    verbose: bool = False,
):
    """Solve eigenvalue problem partially for matrix p."""
    if verbose:
        print(" - Use iterative projector solver.", flush=True)

    if p_block.shape[0] < size_terminate:
        return None, p_block, None

    eigvecs_block = np.zeros(p_block.shape, dtype="double")
    compr = None
    col_id = 0

    for j in range(max_iter):
        if verbose:
            print(" * iteration:", j + 1, flush=True)
        if p_block.shape[0] < size_terminate:
            if verbose:
                print(" iteration stopped. p.shape <", size_terminate, flush=True)
            break

        bool_small = _find_smaller_block(p_block, target_size=3000)
        bool_const = np.logical_not(bool_small)
        n_const = np.count_nonzero(bool_const)
        if verbose:
            print(
                "   - Solving projector of size",
                np.count_nonzero(bool_small),
                flush=True,
            )

        if np.count_nonzero(bool_small) / p_block.shape[0] > 0.95:
            if verbose:
                print(" iteration stopped (> 0.95).", flush=True)
            break

        p_small = p_block[np.ix_(bool_small, bool_small)]
        rank = int(round(np.trace(p_small)))
        if rank == 0:
            break

        eigvecs, (cmplt_eigvals, cmplt_small) = eigh_projector(
            p_small,
            return_complement=True,
            verbose=verbose,
        )
        if eigvecs.shape[1] == 0:
            if verbose:
                print(" No eigenvectors are found.", flush=True)
            break

        compr_size = n_const + cmplt_small.shape[1]
        if verbose:
            print("   - Compressing matrix:", flush=True)
            print("  ", p_block.shape[0], "->", compr_size, flush=True)

        col_end = col_id + eigvecs.shape[1]
        if j == 0:
            compr = np.zeros((p_block.shape[0], compr_size))
            row_ids = np.where(bool_const)[0]
            compr[(row_ids, np.arange(n_const))] = 1.0

            row_ids = np.where(bool_small)[0]
            compr[row_ids, n_const:] = cmplt_small
            eigvecs_block[row_ids, col_id:col_end] = eigvecs
        else:
            compr_slice = compr[:, bool_small]
            prod = compr_slice @ cmplt_small
            eigvecs_block[:, col_id:col_end] = compr_slice @ eigvecs
            """Time consuming part."""
            compr = np.hstack([compr[:, bool_const], prod])
        col_id = col_end

        """
        cmplt_small.T @ p_small @ cmplt_small
        = (cmplt_small.T @ cmplt_small
           @ np.diag(cmplt_eigvals) @ cmplt_small.T @ cmplt_small)
        = np.diag(cmplt_eigvals)
        """
        mat12 = p_block[np.ix_(bool_const, bool_small)] @ cmplt_small
        p_block = np.block(
            [
                [p_block[np.ix_(bool_const, bool_const)], mat12],
                [mat12.T, np.diag(cmplt_eigvals)],
            ]
        )

    if col_id == 0:
        return None, p_block, None
    return eigvecs_block[:, :col_id], p_block, compr


def _iterative_eigsh_projector_use_sparse(
    p_block: np.ndarray,
    max_iter: int = 50,
    size_terminate: int = 2000,
    use_mkl: bool = False,
    verbose: bool = True,
):
    """Solve eigenvalue problem partially for matrix p."""
    if p_block.shape[0] < size_terminate:
        return None, p_block, None

    if verbose:
        print(" - Use iterative projector solver.", flush=True)

    eigvecs_block = np.zeros(p_block.shape, dtype="double")
    compr = None
    col_id = 0

    for j in range(max_iter):
        if verbose:
            print(" * iteration:", j + 1, flush=True)
        if p_block.shape[0] < size_terminate:
            if verbose:
                print(" iteration stopped. p.shape <", size_terminate, flush=True)
            break

        bool_small = _find_smaller_block(p_block, target_size=3000)
        bool_const = np.logical_not(bool_small)
        n_const = np.count_nonzero(bool_const)
        if verbose:
            print(
                "   - Solving projector of size",
                np.count_nonzero(bool_small),
                flush=True,
            )

        if np.count_nonzero(bool_small) / p_block.shape[0] > 0.95:
            if verbose:
                print(" iteration stopped (> 0.95).", flush=True)
            break

        p_small = p_block[np.ix_(bool_small, bool_small)]
        rank = int(round(np.trace(p_small)))
        if rank == 0:
            break

        eigvecs, (cmplt_eigvals, cmplt_small) = eigh_projector(
            p_small,
            return_complement=True,
            verbose=verbose,
        )
        cmplt_small_sp = csr_array(cmplt_small)
        if eigvecs.shape[1] == 0:
            if verbose:
                print(" No eigenvectors are found.", flush=True)
            break

        compr_size = n_const + cmplt_small.shape[1]
        if verbose:
            print("   - Compressing matrix:", flush=True)
            print("  ", p_block.shape[0], "->", compr_size, flush=True)

        col_end = col_id + eigvecs.shape[1]
        if j == 0:
            compr1 = csr_array(
                (np.ones(n_const), (np.where(bool_const)[0], np.arange(n_const))),
                shape=(p_block.shape[0], n_const),
                dtype="double",
            )
            compr2 = lil_array((p_block.shape[0], cmplt_small.shape[1]), dtype="double")
            row_ids = np.where(bool_small)[0]
            compr2[row_ids] = cmplt_small_sp.tolil()
            compr = hstack([compr1, compr2.tocsr()])
            eigvecs_block[row_ids, col_id:col_end] = eigvecs
        else:
            compr_slice = compr[:, bool_small]
            prod = dot_product_sparse(compr_slice, cmplt_small_sp, use_mkl=use_mkl)
            compr = hstack([compr[:, bool_const], prod])
            eigvecs_block[:, col_id:col_end] = compr_slice.toarray() @ eigvecs
        col_id = col_end

        """
        cmplt_small.T @ p_small @ cmplt_small
        = (cmplt_small.T @ cmplt_small
           @ np.diag(cmplt_eigvals) @ cmplt_small.T @ cmplt_small)
        = np.diag(cmplt_eigvals)
        """
        mat12 = p_block[np.ix_(bool_const, bool_small)] @ cmplt_small
        p_block = np.block(
            [
                [p_block[np.ix_(bool_const, bool_const)], mat12],
                [mat12.T, np.diag(cmplt_eigvals)],
            ]
        )

    if col_id == 0:
        return None, p_block, None
    return eigvecs_block[:, :col_id], p_block, compr.toarray()
