"""Generate symmetrized force constants."""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import scipy
from scipy.sparse import coo_array, csr_array

from symfc.matrix_funcs import (
    convert_basis_sets_matrix_form,
    get_projector_constraints,
    get_projector_permutations,
    get_projector_sum_rule,
    kron_c,
    transform_n3n3_serial_to_nn33_serial,
)


class FCBasisSets:
    """Symmetry adapted basis sets for force constants."""

    def __init__(
        self,
        reps: list[coo_array],
        use_exact_projection_matrix: bool = False,
        log_level: int = 0,
        lang: str = "C",
    ):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            Matrix representations of symmetry operations.
        use_exact_projection_matrix : bool, optional
            Use exact projection matrix. Default is False.
        log_level : int, optional
            Log level. Default is 0.
        lang : str, optional
            Compare kron implementations between in python and in C.

        """
        self._reps: list[coo_array] = reps
        self._use_exact_projection_matrix = use_exact_projection_matrix
        self._log_level = log_level

        self._natom = int(round(self._reps[0].shape[0] / 3))

        self._basis_sets: Optional[np.ndarray] = None
        self._fc_basis_seq: Optional[np.ndarray] = None
        self._run(lang=lang)

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

    def _transform_to_fc(self, x):
        fc_b = self._basis_sets.transpose((1, 2, 3, 4, 0))
        return np.dot(fc_b, x)

    def _transform_from_fc(self, fc):
        return np.dot(
            self._fc_basis_seq,
            fc.reshape(
                -1,
            ),
        )

    def _run(self, sparse: bool = True, tol: float = 1e-8, lang: str = "C"):
        proj_R = self._step1(lang=lang)
        nonzero_proj_R = self._step2(proj_R, sparse, tol)
        basis_sets = self._step3(nonzero_proj_R, proj_R, tol)
        self._step4(basis_sets)
        if self._log_level and self._basis_sets is not None:
            print(" fc_basis shape =", self._basis_sets.shape)

    def _step1(self, lang: str = "C") -> csr_array:
        """Construct projection matrix of rotations.

        Returns
        -------
        proj_mat : csr_array
            Projection matrix of rotations in sparse matrix format.
            shape=(3N**2, 3n**2), dtype='double'

        """
        if self._log_level:
            t1 = time.time()
            print(" setting representations for projector ...")

        if lang == "Py":
            row, col, data = self._step1_kron_py()
        elif lang == "Py_for_C":
            row, col, data = self._step1_kron_py_for_c()
        elif lang == "C":
            row, col, data = self._step1_kron_c()

        if self._log_level:
            t2 = time.time()
            print("  - elapsed time =", t2 - t1, "(s)")

        if self._log_level:
            print(" setting sparse projection matrix ...")

        size_sq = (3 * self._natom) ** 2
        proj_mat = csr_array(
            (data, (row, col)), shape=(size_sq, size_sq), dtype="double"
        )

        if self._log_level:
            t3 = time.time()
            print("  - elapsed time =", t3 - t2, "(s)")

        return proj_mat

    def _step1_kron_py(self) -> tuple[list, list, list]:
        """Compute kron(r, r) and reformat from N3N3 into NN33-like."""
        row = []
        col = []
        data = []
        for i, r in enumerate(self._reps):
            print(f"  {i + 1}/{len(self._reps)}")
            sp = scipy.sparse.kron(r, r)
            # serial id in [N,3,N,3] --> serial id in [N,N,3,3]
            row_ids = [
                transform_n3n3_serial_to_nn33_serial(serial, self._natom)
                for serial in sp.row
            ]
            col_ids = [
                transform_n3n3_serial_to_nn33_serial(serial, self._natom)
                for serial in sp.col
            ]
            row += row_ids
            col += col_ids
            data += (sp.data / len(self._reps)).tolist()

        return row, col, data

    def _step1_kron_py_for_c(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute kron(r, r) in NN33 order in Python.

        This is a prototype code to write the same implementation in C.
        See self._step1_kron_c().

                    [a11*B a12*B a13*B ...]
        kron(A, B) =[a21*B a22*B a13*B ...]
                    [a31*B a32*B a33*B ...]
                    [        ...          ]

        (i, j, k, l) N-3-N-3 index
        (i*3+j, k*3+l) N3-N3 index
        (i, k, j, l) N-N-3-3 index
        (i*9*N+k*9+j*3+l) NN33 index

        p = 3*N
        kron(R, R)_(pr+v, ps+w) = R(r,s) * R(v,w)
        i = r // 3
        j = r % 3
        k = s // 3
        l = s % 3
        I = v // 3
        J = v % 3
        K = w // 3
        L = w % 3

        Parameters
        ----------
        rmat : coo_array
            Representation matrix of rotation.
            shape=(N * 3, N * 3), dtype='double'

        """
        size = 0
        for rmat in self._reps:
            size += rmat.row.shape[0] ** 2
        dtype = self._reps[0].row.dtype
        row = np.zeros(size, dtype=dtype)
        col = np.zeros(size, dtype=dtype)
        data = np.zeros(size, dtype=self._reps[0].data.dtype)
        p = 3 * self._natom
        count = 0
        for i, rmat in enumerate(self._reps):
            print(f"  {i + 1}/{len(self._reps)}")
            for r, s, data_l in zip(rmat.row, rmat.col, rmat.data):  # left r
                i_row_R = r // 3
                j_row_R = r % 3
                k_row_R = s // 3
                l_row_R = s % 3
                for v, w, data_r in zip(rmat.row, rmat.col, rmat.data):  # right r
                    i_col_R = v // 3
                    j_col_R = v % 3
                    k_col_R = w // 3
                    l_col_R = w % 3
                    row[count] = i_row_R * 3 * p + i_col_R * 9 + j_row_R * 3 + j_col_R
                    col[count] = k_row_R * 3 * p + k_col_R * 9 + l_row_R * 3 + l_col_R
                    data[count] = data_l * data_r
                    count += 1

        data /= len(self._reps)
        return row, col, data

    def _step1_kron_c(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute kron(r, r) in NN33 order in C.

        See the details about this method in _step1_kron_py_for_c.

        """
        return kron_c(self._reps, self._natom)

    def _step2(
        self, proj_R: Union[csr_array, coo_array], sparse: bool, tol: float
    ) -> np.ndarray:
        """Compute identiy irreps of projection matrix of rotations.

        Select eigenvectors with non-zero eigenvalue (that should be equal to 1) of
        projection matrix of rotations.

        """
        # proj id = 1-dim seq. (N,N,3,3)
        rank = int(round(proj_R.diagonal(k=0).sum()))
        if sparse:
            if self._log_level:
                print(" solving eigenvalue problem for projector")
                print("  (using scipy.sparse.linalg.eigsh) ...")
                print("  - rank (projection matrix) =", rank)
                t1 = time.time()
            vals, vecs = scipy.sparse.linalg.eigsh(proj_R, k=rank, which="LM")
            if self._log_level:
                t2 = time.time()
                print("  - elapsed time =", t2 - t1, "(s)")
        else:
            if self._log_level:
                print(" solving eigenvalue problem for projector")
                print("  (using np.linalg.eigh) ...")
            vals, vecs = np.linalg.eigh(proj_R.toarray())

        nonzero_cols = np.where(np.isclose(vals, 1.0, rtol=0, atol=tol))[0]
        vecs = vecs[:, nonzero_cols]

        if self._log_level:
            print(f" eigenvalues of projector = {vals}")

        return vecs

    def _step3(
        self, nonzero_proj_R: np.ndarray, proj_R: csr_array, tol: float
    ) -> np.ndarray:
        """Construct symmetry adapted basis sets.

        Method-1
        --------
        1. Constraint matrix of sum rule and index permutation symmetry -> C.
        2. Complementary subspace of Ker(C) -> P_c = 1 - C.(C^T.C)^{-1}.C^T.
        3. B' = P_c.B
        4. Select non-zero singular value U (B'=UsV).

        Method-2
        --------
        ...

        """
        if self._log_level:
            print(" applying sum and permutation rules for force constants ...")

        # if nonzero_proj_R.shape[0] < 2:
        if self._use_exact_projection_matrix:
            if self._log_level:
                print("  - using exact projection matrix ... ")

            proj_const = get_projector_constraints(self._natom)

            # checking commutativity of two projectors
            comm = proj_R.dot(proj_const) - proj_const.dot(proj_R)
            if np.any(np.abs(comm.data) > tol):
                raise ValueError("Two projectors do not satisfy commutation rule.")

            U = proj_const.dot(nonzero_proj_R)
        else:
            if self._log_level:
                print("  - using approximate projection matrix ... ")

            proj_sum = get_projector_sum_rule(self._natom)
            proj_perm = get_projector_permutations(self._natom)

            # checking commutativity of two projectors
            comm = proj_R.dot(proj_sum) - proj_sum.dot(proj_R)
            if np.any(np.abs(comm.data) > tol):
                raise ValueError("Two projectors do not satisfy " "commutation rule.")
            comm = proj_R.dot(proj_perm) - proj_perm.dot(proj_R)
            if np.any(np.abs(comm.data) > tol):
                raise ValueError("Two projectors do not satisfy " "commutation rule.")

            n_repeat = 30
            U = proj_sum.dot(nonzero_proj_R)
            for _ in range(n_repeat):
                U = proj_perm.dot(U)
                U = proj_sum.dot(U)

        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print("  - basis size = ", U.shape)

        return U

    def _step4(self, basis_sets: np.ndarray) -> None:
        """Reshape array of symmetry adapted basis sets."""
        if self._log_level:
            t1 = time.time()

        self._fc_basis_seq = basis_sets.T
        fc_basis = [
            b.reshape((self._natom, self._natom, 3, 3)) for b in self._fc_basis_seq
        ]
        self._basis_sets = np.array(fc_basis, dtype="double", order="C")

        if self._log_level:
            t2 = time.time()
            print("  - elapsed time (reshape) =", t2 - t1, "(s)")
