"""Utility functions for eigenvalue solutions."""
import itertools
from collections import defaultdict

import numpy as np
import scipy
from scipy.sparse import csr_array

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


def eigsh_projector(
    p: csr_array, verbose: bool = True, log_interval: int = 10000
) -> csr_array:
    """Solve eigenvalue problem for matrix p.

    Return sparse matrix for eigenvectors of matrix p.

    This algorithm begins with finding block diagonal structures in matrix p.
    For each block submatrix, eigenvectors are solved. When p = diag(A,B), Av =
    v, and Bw = w, p[v,0] = [v,0] and p[0,w] = [0,w] are solutions.

    This function avoids solving numpy.eigh for duplicate block matrices. This
    function is efficient for matrix p composed of many duplicate block
    matrices.

    """
    n_components, labels = scipy.sparse.csgraph.connected_components(p)
    group = defaultdict(list)
    for i, l in enumerate(labels):
        group[l].append(i)

    r = np.array([i for ids in group.values() for i in ids for j in ids])
    c = np.array([j for ids in group.values() for i in ids for j in ids])
    s_end = np.cumsum([len(ids) * len(ids) for ids in group.values()])
    s_begin = np.zeros(len(s_end), dtype=int)
    s_begin[1:] = s_end[:-1]
    p_data = np.ravel(p[r, c])

    if verbose:
        print(" N (blocks) =", n_components)
        rank = int(round(sum(p.diagonal())))
        print(" rank (P) =", rank)

    uniq_eigvecs = dict()
    row, col, data = [], [], []
    col_id = 0
    for i, (s1, s2, ids) in enumerate(zip(s_begin, s_end, group.values())):
        if verbose and (i + 1) % log_interval == 0:
            print(" eigsh_block:", i + 1)

        p_block1 = p_data[s1:s2]
        if len(ids) > 1:
            key = tuple(p_block1)
            try:
                eigvecs = uniq_eigvecs[key]
            except KeyError:
                size = len(ids)
                p_numpy = p_block1.reshape((size, size))
                rank = int(round(np.trace(p_numpy)))
                if rank == 0:
                    eigvals, eigvecs = None, None
                else:
                    eigvals, eigvecs = np.linalg.eigh(p_numpy)
                    nonzero = np.isclose(eigvals, 1.0)
                    eigvecs = eigvecs[:, nonzero]
                uniq_eigvecs[key] = eigvecs
            if eigvecs is not None:
                n_row, n_col = eigvecs.shape
                row.extend([ids[i] for i in range(n_row) for j in range(n_col)])
                col.extend(
                    [j for i in range(n_row) for j in range(col_id, col_id + n_col)]
                )
                data.extend(eigvecs.reshape(-1))
                col_id += n_col
        else:
            if not np.isclose(p_block1[0], 0.0):
                row.append(ids[0])
                col.append(col_id)
                data.append(1.0)
                col_id += 1

    n_col = col_id
    return csr_array((data, (row, col)), shape=(p.shape[0], n_col))


def eigsh_projector_sumrule(p: csr_array, verbose: bool = True) -> np.ndarray:
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
    n_components, labels = scipy.sparse.csgraph.connected_components(p)
    group = defaultdict(list)
    for i, l in enumerate(labels):
        group[l].append(i)

    if verbose:
        print(" n_blocks in P =", n_components)

    eigvecs_full = np.zeros(p.shape, dtype="double")
    col_id = 0
    for i, ids in enumerate(group.values()):
        if verbose:
            print(" eigsh_block:", i, ": block_size =", len(ids))

        p_block = p[np.ix_(ids, ids)].toarray()
        rank = int(round(np.trace(p_block)))
        if rank > 0:
            eigvals, eigvecs = np.linalg.eigh(p_block)
            #eigvals, eigvecs, _ = scipy.linalg.lapack.dsyevd(p_block, 
            #                                                 compute_v=True)
            #eigvals, eigvecs, _, _, _ = scipy.linalg.lapack.dsyevr(p_block, 
            #                                                 compute_v=True)
            nonzero = np.isclose(eigvals, 1.0)
            eigvecs = eigvecs[:, nonzero]
            col_ids = np.arange(col_id, col_id + eigvecs.shape[1])
            eigvecs_full[np.ix_(ids, col_ids)] = eigvecs
            col_id += eigvecs.shape[1]

    return eigvecs_full[:, :col_id]


def connected_components(p: csr_array) -> np.ndarray:
    """Find connected matrix elements.

    This algorithm is simple but inefficient.

    todo: A breadth-first search or depth-first search should be implemented.

    """
    labels = np.arange(p.shape[0], dtype=np.int64)
    for i in range(p.shape[0]):
        if labels[i] == i:
            adj = p.getcol(i).indices
            labels[adj] = i
    return labels


def eigsh_projector_memory_efficient(
    p: csr_array, verbose: bool = True, log_interval: int = 10000
):
    """Eigenvalue solver for connected matrix elements.

    Sparse matrix p must be projector.

    """
    if len(p.data) < 2147483647:
        _, labels = scipy.sparse.csgraph.connected_components(p)
    else:
        labels = connected_components(p)

    group = defaultdict(list)
    for i, l in enumerate(labels):
        group[l].append(i)

    if verbose:
        print(" N (blocks) =", len(group.keys()))
        rank = int(round(sum(p.diagonal())))
        print(" rank (P) =", rank)

    uniq_eigvecs = dict()
    row, col, data = [], [], []
    col_id = 0
    for i, (key, ids) in enumerate(group.items()):
        if verbose and (i + 1) % log_interval == 0:
            print(" eigsh_block:", i + 1)

        if len(ids) > 1:
            r, c = np.array(list(zip(*itertools.product(ids, ids))))
            p_block1 = np.ravel(p[r, c])
            key = tuple(p_block1)
            if key in uniq_eigvecs:
                eigvecs = uniq_eigvecs[key]
            else:
                size = len(ids)
                p_numpy = p_block1.reshape((size, size))
                rank = int(round(np.trace(p_numpy)))
                if rank == 0:
                    eigvals, eigvecs = None, None
                else:
                    eigvals, eigvecs = np.linalg.eigh(p_numpy)
                    nonzero = np.isclose(eigvals, 1.0)
                    eigvecs = eigvecs[:, nonzero]
                uniq_eigvecs[key] = eigvecs

            if eigvecs is not None:
                n_row, n_col = eigvecs.shape
                row.extend([ids[i] for i in range(n_row) for j in range(n_col)])
                col.extend(
                    [j for i in range(n_row) for j in range(col_id, col_id + n_col)]
                )
                data.extend(eigvecs.reshape(-1))
                col_id = n_col
        else:
            if not np.isclose(p[ids[0], ids[0]], 0.0):
                row.append(ids[0])
                col.append(col_id)
                data.append(1.0)
                col_id += 1

    n_col = col_id
    return csr_array((data, (row, col)), shape=(p.shape[0], n_col))
