"""Generate symmetrized force constants using compact projection matrix."""
from typing import Literal, Optional

import numpy as np
import scipy
from phonopy.structure.atoms import PhonopyAtoms
from scipy.sparse import coo_array

from symfc.spg_reps import SpgReps
from symfc.utils import (
    get_indep_atoms_by_lat_trans,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
    get_spg_projector,
)


class FCBasisSet:
    """Symmetry adapted basis set for force constants.

    Attributes
    ----------
    basis_set : ndarray
        Force constants basis set.
        shape=(n_a * N * 9, n_basis), dtype='double'

    """

    def __init__(
        self,
        supercell: PhonopyAtoms,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._spg_reps = SpgReps(supercell).run()
        self._natom = len(supercell)
        self._log_level = log_level
        self._basis_set: Optional[np.ndarray] = None
        self._mode = Literal["lowmem"]
        self._compression_mat = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return basis set in (n_a * N * 9, n_basis) array."""
        return self._basis_set

    @property
    def decompression_indices(self):
        """Return decompression indices in (N,N,3,3) order.

        shape=(N^2*9,), dtype='int_'.

        """
        trans_perms = self._spg_reps.translation_permutations
        return get_lat_trans_decompr_indices(trans_perms)

    @property
    def compression_indices(self):
        """Return compression indices in (n_a,N,3,3) order.

        shape=(n_a*N*9, n_lp), dtype='int_'.

        """
        trans_perms = self._spg_reps.translation_permutations
        return get_lat_trans_compr_indices(trans_perms)

    @property
    def translation_permutations(self):
        """Return permutations by lattice translation.

        Returns
        --------
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int

        """
        return self._spg_reps.translation_permutations

    def run(self, mode: Literal["fast", "lowmem"] = "lowmem", tol: float = 1e-8):
        """Compute force constants basis.

        Parameters
        ----------
        mode : Lietral["fast", "lowmem"]
            With "fact", basis set is computed faster but with more memory usage
            and with "lowmem" slower but less memory usage. Default is "fast".

        """
        self._mode = mode
        compression_mat = get_lat_trans_compr_matrix(
            self._spg_reps.translation_permutations
        )
        vecs = self._step1(compression_mat, tol=tol)
        U = self._step2(vecs, compression_mat)
        self._step3(U, tol=tol)
        return self

    def solve(
        self, displacements: np.ndarray, forces: np.ndarray, is_compact_fc=True
    ) -> Optional[np.ndarray]:
        """Solve force constants.

        Parameters
        ----------
        displacements : ndarray
            Displacements of atoms in Cartesian coordinates. shape=(n_snapshot,
            N, 3), dtype='double'
        forces : ndarray
            Forces of atoms in Cartesian coordinates. shape=(n_snapshot, N, 3),
            dtype='double'
        is_compact_fc : bool
            Shape of force constants array is (n_a, N, 3, 3) if True or (N, N,
            3, 3) if False.

        Returns
        -------
        ndarray
            Force constants.
            shape=(n_a, N, 3, 3) or (N, N, 3, 3). See `is_compact_fc` parameter.
            dtype='double', order='C'

        """
        if self._basis_set is None:
            return None
        assert displacements.shape == forces.shape
        coeff = self._get_basis_coeff(displacements, forces)
        trans_perms = self._spg_reps.translation_permutations
        N = self._natom
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        fc = (self._basis_set @ coeff)[decompr_idx].reshape(N, N, 3, 3)
        if is_compact_fc:
            indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
            return np.array(fc[indep_atoms], dtype="double", order="C")
        return fc

    def _get_basis_coeff(
        self,
        displacements: np.ndarray,
        forces: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._basis_set is None:
            return None
        trans_perms = self._spg_reps.translation_permutations
        decompr_idx = get_lat_trans_decompr_indices(trans_perms, shape="N3,N3")
        n_snapshot = displacements.shape[0]
        disps = displacements.reshape(n_snapshot, -1)
        N = self._natom
        d_basis = np.zeros(
            (n_snapshot * N * 3, self._basis_set.shape[1]), dtype="double", order="C"
        )
        for i, vec in enumerate(self._basis_set.T):
            d_basis[:, i] = (disps @ vec[decompr_idx]).ravel()
        coeff = -(np.linalg.pinv(d_basis) @ forces.ravel())
        return coeff

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
            N = self._natom**2 * 9
            N_c = compression_spg_mat.shape[0]
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
            return self._step2_lowmem(vecs)
        else:
            raise RuntimeError("This should not happen.")

    def _step2_lowmem(self, vecs: np.ndarray) -> np.ndarray:
        if self._log_level:
            print("Multiply sum rule projector with current basis set...")
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape
        n_a = N // n_lp
        U = np.zeros(shape=(n_a * 9 * N, vecs.shape[1]), dtype="double")
        compr_idx = get_lat_trans_compr_indices(trans_perms)
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        for i, vec in enumerate(vecs.T):
            U[:, i] = self._get_U_basis(vec, compr_idx, decompr_idx)
        return U

    def _get_U_basis(
        self,
        vec: np.ndarray,
        compr_idx: np.ndarray,
        decompr_idx: np.ndarray,
    ):
        N = self._natom
        basis = vec[decompr_idx].reshape(N, N, 9).sum(axis=1)
        basis = np.tile(basis, N).ravel()
        basis = basis[compr_idx].sum(axis=1) / (compr_idx.shape[1] * N)
        return vec - basis

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

    def _step3(self, U: np.ndarray, tol: float = 1e-8):
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
            print(
                "Non-zero elems: "
                f"{np.count_nonzero(np.abs(U) > tol)}/{np.prod(U.shape)}"
            )

        self._basis_set = U
