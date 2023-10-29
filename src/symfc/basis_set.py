"""Generate symmetrized force constants using compact projection matrix."""
import time
from typing import Literal, Optional

import numpy as np
import scipy
from scipy.sparse import coo_array

from symfc.spg_reps import SpgReps
from symfc.utils import (
    convert_basis_set_matrix_form,
    get_indep_atoms_by_lat_trans,
    get_lat_trans_compr_matrix,
    get_lat_trans_compr_matrix_block_i,
    get_spg_projector,
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

    def __init__(
        self,
        spg_reps: SpgReps,
        mode: Literal["fast", "lowmem"] = "fast",
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        reps : list[coo_array]
            3Nx3N matrix representations of symmetry operations.
        translation_permutations:
            Atom indices after lattice translations.
            shape=(lattice_translations, supercell_atoms)
        mode : Lietral["fast", "lowmem"]
            With "fact", basis set is computed faster but with more memory usage
            and with "lowmem" slower but less memory usage. Default is "fast".
        log_level : int, optional
            Log level. Default is 0.

        """
        self._spg_reps = spg_reps
        self._mode = mode
        self._natom = len(self._spg_reps.numbers)
        self._log_level = log_level
        self._basis_set: Optional[np.ndarray] = None

    def basis_set_matrix_form(self) -> Optional[list[np.ndarray]]:
        """Retrun basis set in (n_basis, 3N, 3N) array."""
        return convert_basis_set_matrix_form(self.fc_basis_set)

    @property
    def fc_basis_set(self) -> Optional[np.ndarray]:
        """Return basis set in (n_basis, N, N, 3, 3) array."""
        return self._get_fc_basis_set()

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return basi set in (n_a * N * 9, n_basis) array."""
        return self._basis_set

    @property
    def compression_matrix(self) -> coo_array:
        """Return compression matrix."""
        return get_lat_trans_compr_matrix(self._spg_reps.translation_permutations)

    def run(self, tol: float = 1e-8):
        """Compute force constants basis."""
        if self._log_level:
            print("Construct compression matrix of lattice translation.")
        compression_mat = self.compression_matrix
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
                "Construct projector matrix of space group and "
                "index permutation symmetry..."
            )
        compression_spg_mat = get_spg_projector(
            self._spg_reps,
            self._natom,
            compression_mat,
        )
        rank = int(round(compression_spg_mat.diagonal(k=0).sum()))
        if self._log_level:
            N, N_c = compression_mat.shape
            print(f"Projection matrix ({N}, {N}) was compressed to ({N_c}, {N_c}).")
            print(
                f"Solving eigenvalue problem of projection matrix with rank={rank}..."
            )
        vals, vecs = scipy.sparse.linalg.eigsh(compression_spg_mat, k=rank, which="LM")
        # Check non-zero values are all ones. This is a weak check of
        # commutativity.
        np.testing.assert_allclose(vals, 1.0, rtol=0, atol=tol)
        return vecs

    def _step2(self, vecs: np.ndarray, compression_mat: coo_array) -> np.ndarray:
        if self._mode == "fast":
            return self._step2_fast(vecs, compression_mat)
        elif self._mode == "lowmem":
            return self._step2_lowmem(vecs, compression_mat)
        else:
            raise RuntimeError("This should not happen.")

    def _step2_lowmem(self, vecs: np.ndarray, compression_mat: coo_array) -> np.ndarray:
        if self._log_level:
            print("Multiply sum rule projector with current basis set...")
        n_l, N = self._spg_reps.translation_permutations.shape
        n_a = N // n_l
        U_sum = np.zeros(shape=(n_a, 9 * N, vecs.shape[1]), dtype="double")
        for i_lattice in range(n_l):
            U_sum += self._get_U_i_lat(i_lattice, n_a, vecs)
        return U_sum.reshape(n_a * 9 * N, -1)

    def _get_U_i_lat(
        self,
        i_lattice: int,
        n_a: int,
        vecs: np.ndarray,
    ):
        if self._log_level:
            print(f"---------{i_lattice}---------")
        t0 = time.time()
        compr_block = get_lat_trans_compr_matrix_block_i(
            self._spg_reps.translation_permutations, i_lattice
        )
        if self._log_level:
            print("compr_block", compr_block.shape, time.time() - t0)
        t1 = time.time()
        U = compr_block @ vecs.reshape(n_a, self._natom * 9, vecs.shape[1])
        t2 = time.time()
        if self._log_level:
            print("compr_block @ vecs", U.shape, t2 - t1)
        prod = U.reshape(n_a, self._natom, 9, -1).sum(axis=1) / self._natom
        t3 = time.time()
        if self._log_level:
            print("sum", prod.shape, t3 - t2)
        for j in range(self._natom):
            U[:, j * 9 : (j + 1) * 9, :] -= prod
        t4 = time.time()
        if self._log_level:
            print("loop", t4 - t3)
        U = compr_block.T @ U
        t5 = time.time()
        if self._log_level:
            print("compr_block.T @ U", U.shape, t5 - t4)
        return U

    def _step2_fast(self, vecs: np.ndarray, compression_mat: coo_array) -> np.ndarray:
        """Multiply sum rule project to basis vectors.

        Compute P_sum B, where

        P_sum = I - M_sum = I - np.kron(np.eye(natom, natom),
                                      np.tile(np.eye(9) / natom, (natom,
                                      natom)))

        and B is the set of fc basis vectors determined by P_SG and P_perm.

        """
        if self._log_level:
            print("Multiply sum rule projector with current basis set...")
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
        if self._log_level:
            print("Accomodate sum rule by SVD...")

        U, s, _ = np.linalg.svd(U, full_matrices=False)
        U = U[:, np.where(np.abs(s) > 1 - tol)[0]]

        if self._log_level:
            print("Excluded SVD eigenvalues:")
            print(f"{s[np.abs(s) < 1 - tol]}")
            print(f"Final size of basis set: {U.shape}")

        self._basis_set = U

    def _get_compact_fc_basis_set(self) -> Optional[np.ndarray]:
        if self._basis_set is None:
            return None

        trans_perms = self._spg_reps.translation_permutations
        n_l, N = trans_perms.shape
        n_a = N // n_l
        # fc_basis_set = np.zeros(
        #     (self._basis_set.shape[1], len(indep_atoms), N, 3, 3), dtype="double"
        # )
        compr_block = get_lat_trans_compr_matrix_block_i(
            self._spg_reps.translation_permutations, 0
        )
        basis_part = compr_block @ self._basis_set.reshape(n_a, N * 9, -1)
        fc_basis_set = basis_part.reshape(n_a * N * 9, -1).T.reshape(-1, n_a, N, 3, 3)
        return fc_basis_set

    def _get_fc_basis_set(self) -> Optional[np.ndarray]:
        if self._basis_set is None:
            return None
        if self._mode == "lowmem":
            trans_perms = self._spg_reps.translation_permutations
            indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
            n_l, N = trans_perms.shape
            n_a = N // n_l
            fc_basis_set = np.zeros(
                (self._basis_set.shape[1], N, N, 3, 3), dtype="double"
            )
            for i_lattice in range(n_l):
                if self._log_level:
                    print(f"{i_lattice}")
                compr_block = get_lat_trans_compr_matrix_block_i(
                    self._spg_reps.translation_permutations, i_lattice
                )
                basis_part = compr_block @ self._basis_set.reshape(n_a, N * 9, -1)
                fc_basis_set[
                    :, trans_perms[i_lattice, indep_atoms], :, :, :
                ] = basis_part.reshape(n_a * N * 9, -1).T.reshape(-1, n_a, N, 3, 3)
            return fc_basis_set
        elif self._mode == "fast":
            fc_basis_set = (self.compression_matrix @ self._basis_set).T.reshape(
                (self._basis_set.shape[1], self._natom, self._natom, 3, 3)
            )
            return fc_basis_set
        else:
            raise RuntimeError("This should not happen.")
