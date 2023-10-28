"""Generate symmetrized force constants using compact projection matrix."""
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import coo_array

from symfc.spg_reps import SpgReps
from symfc.utils import (
    convert_basis_set_matrix_form,
    get_compression_spg_proj,
    get_lattice_translation_compression_matrix,
)


class FCBasisSet:
    """Symmetry adapted basis set for force constants.

    Strategy
    --------
    Construct compression matrix using lattice translation symmetry C. The
    matrix shape is (NN33, n_aN33) where N=n_a * n_l. n_a and n_l is the number
    of atoms in primitive cell and the number of lattice points in the
    supercell. This matrix expands elements of full elements NN33 of matrix.
    (C.T @ C) is made to be identity matrix. The projection matrix of space
    group operations is multipiled by C from both side, and the resultant matrix
    is diagonalized. In addition, (C_perm @ C_perm.T) that is the projector of
    index permutation symmetry is multiplied.

    """

    def __init__(self, spg_reps: SpgReps, log_level: int = 0):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            3Nx3N matrix representations of symmetry operations.
        translation_permutations:
            Atom indices after lattice translations.
            shape=(lattice_translations, supercell_atoms)
        log_level : int, optional
            Log level. Default is 0.

        """
        self._spg_reps = spg_reps
        # self._reps: list[coo_array] = spg_reps.representations
        self._natom = len(self._spg_reps.numbers)
        self._log_level = log_level
        self._basis_set: Optional[np.ndarray] = None

    @property
    def basis_set_matrix_form(self) -> Optional[list[np.ndarray]]:
        """Retrun a list of FC basis in 3Nx3N matrix."""
        if self._basis_set is None:
            return None

        return convert_basis_set_matrix_form(self._basis_set)

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return a list of FC basis in (N, N, 3, 3) dimentional arrays."""
        return self._basis_set

    def run(
        self,
        tol: float = 1e-8,
    ):
        """Compute force constants basis."""
        if self._log_level:
            print("Construct compression matrix of lattice translation.")
        compression_mat = get_lattice_translation_compression_matrix(
            self._spg_reps.translation_permutations
        )
        vecs = self._step1(compression_mat, tol=tol)
        U = self._step2(vecs, compression_mat)
        self._step3(U, compression_mat, tol=tol)
        return self

    def _step1(
        self,
        compression_mat: coo_array,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """Compute eigenvectors of projection matrix.

        Projection matrix is made of the product of the projection matrices of
        space group operations and index permutation symmetry in supercell.

        The eigenvalues are 1 or 0. Therefore eigenvectors corresponding to
        eigenvalue=1 are collected using sparce eigen solver. The collected
        eigenvectors are basis vectors of force constants.

        """
        if self._log_level:
            print(
                "Construct projector of product of space group and "
                "index permutation symmetry."
            )
        compression_spg_mat = get_compression_spg_proj(
            self._spg_reps,
            self._natom,
            compression_mat,
        )
        rank = int(round(compression_spg_mat.diagonal(k=0).sum()))
        if self._log_level:
            print(f"Solving eigenvalue problem of projection matrix (rank={rank}).")
        vals, vecs = scipy.sparse.linalg.eigsh(compression_spg_mat, k=rank, which="LM")
        nonzero_elems = np.nonzero(np.abs(vals) > tol)[0]
        vals = vals[nonzero_elems]
        # Check non-zero values are all ones. This is a weak check of
        # commutativity.
        np.testing.assert_allclose(vals, 1.0, rtol=0, atol=tol)
        vecs = vecs[:, nonzero_elems]
        if self._log_level:
            print(f" eigenvalues of projector = {vals}")
        return vecs

    def _step2(self, vecs: np.ndarray, compression_mat: coo_array) -> np.ndarray:
        """Multiply sum rule project to basis vectors.

        Compute P_sum B, where

        P_sum = I - M_sum = I - np.kron(np.eye(natom, natom),
                                      np.tile(np.eye(9) / natom, (natom,
                                      natom)))

        and B is the set of fc basis vectors determined by P_SG and P_perm.

        """
        print("Multiply sum rule projector")
        U = compression_mat @ vecs
        for i in range(self._natom):
            prod = (
                U[i * self._natom * 9 : (i + 1) * self._natom * 9, :]
                .reshape(self._natom, 9, -1)
                .sum(axis=0)
                / self._natom
            )
            for j in range(self._natom):
                U[(i * self._natom + j) * 9 : (i * self._natom + j + 1) * 9, :] -= prod
        U = compression_mat.T @ U
        return U

    def _step3(self, U: np.ndarray, compression_mat: coo_array, tol: float = 1e-8):
        """Extract basis vectors that satisfies sum rule.

        Eigenvectors corresponding to SVD eigenvalues that are not 1 are
        rejected.

        """
        print("Solving SVD")
        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > 1 - tol)[0]]

        if self._log_level:
            print(f"  - svd eigenvalues = {np.abs(s)}")
            print(f"  - basis size = {U.shape}")

        self._basis_set = (compression_mat @ U).T.reshape(
            (U.shape[1], self._natom, self._natom, 3, 3)
        )
