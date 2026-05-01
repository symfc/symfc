"""Functions for graph."""

import numpy as np
from scipy.sparse import csr_array, issparse


def get_cols(p: csr_array, node: int):
    """Return column indices of csr array."""
    start = p.indptr[node]
    end = p.indptr[node + 1]
    if start == 0:
        return p.indices[end - 1 :: -1]
    return p.indices[end - 1 : start - 1 : -1]


def dfs_tree_stack(p: csr_array, visited: np.ndarray, initial_node: int = 0):
    """Run depth first search using stack for sparse matrix p."""
    visited_nodes = []
    stack = [initial_node]

    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            visited_nodes.append(node)
            cols = get_cols(p, node)
            unvisited = cols[~visited[cols]]
            stack.extend(unvisited.tolist())
    return visited, sorted(visited_nodes)


def connected_components(p: csr_array):
    """Calculate connected components of graph p."""
    if not issparse(p):
        raise RuntimeError("Not sparse matrix.")

    p.eliminate_zeros()
    visited = np.zeros(p.shape[0], dtype=bool)

    group, label = dict(), 0
    for node in range(p.shape[0]):
        if visited[node]:
            continue

        visited, visited_nodes = dfs_tree_stack(p, visited, initial_node=node)
        group[label] = visited_nodes
        label += 1

    return group
