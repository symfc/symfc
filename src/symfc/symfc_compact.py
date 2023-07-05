"""Generate symmetrized force constants using compact projection matrix."""
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array

from symfc.matrix_funcs import convert_basis_sets_matrix_form, kron_c, to_serial
from symfc.symfc import get_projector_constraints


class SymBasisSetsCompact:
    """Compact symmetry adapted basis sets for force constants."""

    def __init__(
        self,
        reps: list[coo_array],
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            Matrix representations of symmetry operations.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._reps: list[coo_array] = reps
        self._log_level = log_level

        self._natom = int(round(self._reps[0].shape[0] / 3))

        self._basis_sets: Optional[np.ndarray] = None

        self._run()

    @property
    def basis_sets_matrix_form(self) -> Optional[list[np.ndarray]]:
        """Retrun a list of FC basis in 3Nx3N matrix."""
        if self._basis_sets is None:
            return None

        return convert_basis_sets_matrix_form(self._basis_sets)

    @property
    def basis_sets(self) -> Optional[np.ndarray]:
        """Return a list of FC basis in (N, N, 3, 3) dimentional arrays."""
        return self._basis_sets

    def _run(self, tol: float = 1e-8):
        row, col, data = kron_c(self._reps, self._natom)
        size_sq = (3 * self._natom) ** 2
        proj_mat = coo_array(
            (data, (row, col)), shape=(size_sq, size_sq), dtype="double"
        )
        perm_mat = _get_permutation_compression_matrix(self._natom)

        perm_proj_mat = (perm_mat @ proj_mat) @ perm_mat.T
        rank = int(round(perm_proj_mat.diagonal(k=0).sum()))
        print("Solving eigenvalue problem of projection matrix.")
        vals, vecs = scipy.sparse.linalg.eigsh(perm_proj_mat, k=rank, which="LM")
        # vals, vecs = np.linalg.eigh(perm_proj_mat.toarray())
        nonzero_cols = np.where(np.isclose(vals, 1.0, rtol=0, atol=tol))[0]
        vecs = vecs[:, nonzero_cols]
        if self._log_level:
            print(
                f" eigenvalues of projector = {vals}, len(nonzero-vals)={vecs.shape[1]}"
            )
        proj_const = get_projector_constraints(self._natom, with_permutation=False)

        # checking commutativity of two projectors
        comm = proj_mat.dot(proj_const) - proj_const.dot(proj_mat)
        if np.all(np.abs(comm.data) < tol) is False:
            raise ValueError("Two projectors do not satisfy commutation rule.")
        U = proj_const.dot(perm_mat.T @ vecs)
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print("  - basis size = ", U.shape)

        fc_basis = [b.reshape((self._natom, self._natom, 3, 3)) for b in U.T]
        self._basis_sets = np.array(fc_basis, dtype="double", order="C")


def _get_permutation_compression_matrix(natom: int) -> coo_array:
    """Return compression matrix by permutation matrix.

    Matrix shape is ((N*3)((N*3)+1)/2, NN33).
    Non-zero only ijab and jiba column elements for ijab rows.
    Rows upper right NN33 matrix elements are selected for rows.

    """
    row, col, data = [], [], []
    size = natom**2 * 9

    count = 0
    for i_i in range(natom):
        for i_j in range(natom):
            if i_i > i_j:
                continue
            for i_a in range(3):
                for i_b in range(3):
                    if i_i == i_j and i_a > i_b:
                        continue
                    row.append(count)
                    col.append(to_serial(i_i, i_a, i_j, i_b, natom))
                    if i_i == i_j and i_a == i_b:
                        data.append(1)
                    else:
                        data.append(np.sqrt(2) / 2)
                        row.append(count)
                        col.append(to_serial(i_j, i_b, i_i, i_a, natom))
                        data.append(np.sqrt(2) / 2)
                    count += 1
    if (natom * 3) % 2 == 1:
        assert (natom * 3) * ((natom * 3 + 1) // 2) == count, f"{natom}, {count}"
    else:
        assert ((natom * 3) // 2) * (natom * 3 + 1) == count, f"{natom}, {count}"
    return coo_array((data, (row, col)), shape=(count, size), dtype="double")
