"""Symmetry adapted basis sets of 2nd order force constants."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO2
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix_tools_O2 import compressed_projector_sum_rules
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O2 import (
    get_compr_coset_reps_sum,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
    get_perm_compr_matrix,
    get_spg_perm_projector,
)

from .basis_sets_base import FCBasisSetBase


class FCBasisSetO2Base(FCBasisSetBase):
    """Base class of FCBasisSetO2."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        use_mkl : bool
            Use MKL or not. Default is False.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO2(supercell)

    def _get_c_trans(self) -> csr_array:
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
        return c_trans


class FCBasisSetO2Slow(FCBasisSetO2Base):
    """Symmetry adapted basis set for 2nd order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set.
        shape=(n_a * N * 9, n_bases), dtype='double'
    full_basis_set : ndarray
        Full (decompressed) force constants basis set.
        shape=(N * N * 9, n_bases), dtype='double'
    decompression_indices : ndarray
        Decompression indices in (N,N,3,3) order.
        shape=(N^2*9,), dtype='int_'.
    compresssion_indices : ndarray
        Compression indices in (n_a,N,3,3) order.
        shape=(n_a*N*9, n_lp), dtype='int_'.
    translation_permutations : ndarray
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int.

    """

    def __init__(
        self,
        supercell: SymfcAtoms,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, log_level=log_level)
        self._spg_reps = SpgRepsO2(supercell)

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        shape=(n_a*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_a,N,3,3).

        """
        return self._basis_set

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        """Return full (decompressed) basis set.

        shape=(N*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (N,N,3,3).

        """
        if self._basis_set is None:
            return None
        return self._basis_set[self.decompression_indices, :]

    @property
    def compact_compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix.

        This expands fc basis_sets to (n_a*N*3*3, n_bases).

        """
        return 1

    @property
    def compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix.

        This expands fc basis_sets to (N*N*3*3, n_bases).

        """
        n_lp = self.translation_permutations.shape[0]
        return self._get_c_trans() * np.sqrt(n_lp)

    @property
    def decompression_indices(self) -> np.ndarray:
        """Return decompression indices in (N,N,3,3) order."""
        trans_perms = self.translation_permutations
        return get_lat_trans_decompr_indices(trans_perms)

    @property
    def compression_indices(self) -> np.ndarray:
        """Return compression indices in (n_a,N,3,3) order."""
        return get_lat_trans_compr_indices(self.translation_permutations)

    def run(self, tol: float = 1e-8) -> FCBasisSetO2Slow:
        """Compute compressed force constants basis set.

        Parameters
        ----------
        tol : float
            Tolerance to identify zero eigenvalues. Default=1e-8.

        """
        decompr_idx = get_lat_trans_decompr_indices(self.translation_permutations)
        vecs = self._get_tilde_basis_set(decompr_idx, tol=tol)
        U = self._multiply_sum_rule_projector(decompr_idx, vecs)
        self._extract_basis_set(U, tol=tol)
        return self

    def _get_tilde_basis_set(
        self, decompr_idx: np.ndarray, tol: float = 1e-8
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
        compression_spg_mat = get_spg_perm_projector(self._spg_reps, decompr_idx)
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

    def _multiply_sum_rule_projector(
        self, decompr_idx: np.ndarray, vecs: np.ndarray
    ) -> np.ndarray:
        if self._log_level:
            print("Multiply sum rule projector with current basis set...")
        n_lp, N = self.translation_permutations.shape
        n_a = N // n_lp
        U = np.zeros(shape=(n_a * 9 * N, vecs.shape[1]), dtype="double")
        compr_idx = self.compression_indices
        for i, vec in enumerate(vecs.T):
            basis = vec[decompr_idx].reshape(N, N, 9).sum(axis=1)
            basis = np.tile(basis, N).ravel()
            basis = basis[compr_idx].sum(axis=1) / (compr_idx.shape[1] * N)
            U[:, i] = vec - basis
        return U

    def _extract_basis_set(self, U: np.ndarray, tol: float = 1e-8):
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


class FCBasisSetO2(FCBasisSetO2Base):
    """Dense symmetry adapted basis set for 2nd order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set. The first dimension n_x (< n_a) is
        given as a result of compression, which is depends on the system.
        shape=(n_x * N * 9, n_bases), dtype='double'
    full_basis_set : ndarray
        Full (decompressed) force constants basis set. shape=(N * N * 9,
        n_bases), dtype='double'
    translation_permutations : ndarray
        Atom indices after lattice translations. shape=(lattice_translations,
        supercell_atoms), dtype=int.

    """

    def __init__(
        self,
        supercell: SymfcAtoms,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO2(supercell)
        self._n_a_compression_matrix: Optional[csr_array] = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        n_c = len(compressed_indices).

        shape=(n_c*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_c,N,3,3).

        """
        return self._basis_set

    @property
    def compact_basis_set(self) -> Optional[np.ndarray]:
        """Return compact basis set.

        n_a : number of atoms in primitive cell.

        shape=(n_a*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_a,N,3,3).

        """
        if self._basis_set is None:
            return None
        return dot_product_sparse(
            self._n_a_compression_matrix, self._basis_set, use_mkl=self._use_mkl
        )

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        """Return full (decompressed) basis set.

        shape=(N*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (N,N,3,3).

        """
        if self._basis_set is None:
            return None
        return dot_product_sparse(
            self.compression_matrix, self._basis_set, use_mkl=self._use_mkl
        )

    @property
    def compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix.

        This expands fc basis_sets to (N*N*3*3, n_bases).

        """
        c_trans = self._get_c_trans()
        return dot_product_sparse(
            c_trans, self._n_a_compression_matrix, use_mkl=self._use_mkl
        )

    @property
    def compact_compression_matrix(self) -> Optional[csr_array]:
        """Return compact compression matrix.

        This expands basis_sets to (n_a*N*3*3, n_bases).

        """
        n_lp = self.translation_permutations.shape[0]
        return self._n_a_compression_matrix / np.sqrt(n_lp)

    def run(self) -> FCBasisSetO2:
        """Compute compressed force constants basis set.

        compression using C(pt)
            = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
        proj_rpt = c_pt.transpose() @ coset_reps_sum @ c_pt

        [C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt] @ C_rpt
         = C_rpt @ [C_rpt.T @ C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt @ C_rpt]
         = C_rpt @ compression_mat.T @ (I - P_sum^(c)) @ compression_mat
         = C_rpt @ compression_mat.T @ (I - Csum(c) @ Csum(c).T) @ compression_mat
         = C_rpt @ proj

        compress_mat = c_trans @ c_pt @ c_rpt

        """
        N = self._natom
        c_trans = self._get_c_trans()
        n_a_compress_mat = self._get_n_a_compress_mat(c_trans)
        compress_mat = dot_product_sparse(
            c_trans, n_a_compress_mat, use_mkl=self._use_mkl
        )
        proj = compressed_projector_sum_rules(compress_mat, N, use_mkl=self._use_mkl)
        eigvecs = eigsh_projector_sumrule(proj)

        if self._log_level:
            print(f"Final size of basis set: {eigvecs.shape}")

        self._basis_set = eigvecs
        self._n_a_compression_matrix = n_a_compress_mat
        return self

    def _get_n_a_compress_mat(self, c_trans: csr_array) -> csr_array:
        """Return compression matrix without c_trans mutiplied.

        This compression matrix is preserved as a class instance variable.
        The full compression matrix is obtained by

        c_trans @ n_a_compression_matrix.

        The compact compression matrix is obtained by

        n_a_compression_matrix / sqrt(n_lp).

        """
        N = self._natom
        c_perm = get_perm_compr_matrix(N)
        c_pt = dot_product_sparse(c_perm.T, c_trans, use_mkl=self._use_mkl)
        proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=self._use_mkl)
        coset_reps_sum = get_compr_coset_reps_sum(self._spg_reps)
        c_pt = eigsh_projector(proj_pt)
        proj_rpt = dot_product_sparse(coset_reps_sum, c_pt, use_mkl=self._use_mkl)
        proj_rpt = dot_product_sparse(c_pt.T, proj_rpt)
        c_rpt = eigsh_projector(proj_rpt)
        n_a_compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=self._use_mkl)
        return n_a_compress_mat
