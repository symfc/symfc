"""Symmetry adapted basis sets of 2nd order force constants."""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import scipy
from phonopy.structure.atoms import PhonopyAtoms
from scipy.sparse import coo_array, csr_array

from symfc.spg_reps import SpgRepsO2
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix_tools_O2 import compressed_projector_sum_rules
from symfc.utils.utils_O2 import (
    get_compr_coset_reps_sum,
    get_lat_trans_compr_indices,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
    get_perm_compr_matrix,
    get_spg_perm_projector,
)

from .basis_sets_base import FCBasisSetBase


def print_sp_matrix_size(c: Union[csr_array, coo_array], header: str):
    """Show sparse matrix size."""
    print(header, c.shape, len(c.data))


class FCBasisSetO2Slow(FCBasisSetBase):
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


class FCBasisSetO2(FCBasisSetBase):
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
        supercell: PhonopyAtoms,
        use_mkl: bool = False,
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
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO2(supercell)

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        shape=(n_x*N*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_x,N,3,3).

        n_x (< n_a) is given as a result of compression, which is depends on the
        system.

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
    def compression_matrix(self) -> Optional[csr_array]:
        """Return compression matrix."""
        return self._compression_matrix

    def run(self):
        """Compute compressed force constants basis set."""
        tt_begin = time.time()
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape

        tt00 = time.time()
        """C(permutation)"""
        c_perm = get_perm_compr_matrix(N)
        print_sp_matrix_size(c_perm, " C_perm:")
        tt0 = time.time()

        """C(trans)"""
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
        print_sp_matrix_size(c_trans, " C_trans:")
        tt1 = time.time()

        """C(pt) = C(perm).T @ C(trans)"""
        c_pt = dot_product_sparse(c_perm.T, c_trans, use_mkl=self._use_mkl)
        print_sp_matrix_size(c_pt, " C_(perm,trans):")
        proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=self._use_mkl)
        print_sp_matrix_size(proj_pt, " P_(perm,trans):")
        tt2 = time.time()

        coset_reps_sum = get_compr_coset_reps_sum(self._spg_reps)
        print_sp_matrix_size(coset_reps_sum, " R_(coset):")
        tt3 = time.time()

        """
        compression using C(pt)
            = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
        proj_rpt = c_pt.transpose() @ coset_reps_sum @ c_pt
        """
        c_pt = eigsh_projector(proj_pt)
        print_sp_matrix_size(c_pt, " C_(perm,trans,compressed):")

        proj_rpt = dot_product_sparse(coset_reps_sum, c_pt, use_mkl=self._use_mkl)
        proj_rpt = dot_product_sparse(c_pt.T, proj_rpt)
        print_sp_matrix_size(proj_rpt, " P_(perm,trans,coset):")
        tt4 = time.time()

        c_rpt = eigsh_projector(proj_rpt)
        print_sp_matrix_size(c_rpt, " C_(perm,trans,coset):")
        tt5 = time.time()

        """
        [C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt] @ C_rpt
         = C_rpt @ [C_rpt.T @ C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt @ C_rpt]
         = C_rpt @ compression_mat.T @ (I - P_sum^(c)) @ compression_mat
         = C_rpt @ compression_mat.T @ (I - Csum(c) @ Csum(c).T) @ compression_mat
         = C_rpt @ proj
            compress_mat = c_trans @ c_pt @ c_rpt
        """
        compress_mat = dot_product_sparse(c_trans, c_pt, use_mkl=self._use_mkl)
        compress_mat = dot_product_sparse(compress_mat, c_rpt, use_mkl=self._use_mkl)
        print_sp_matrix_size(compress_mat, " compression matrix:")

        proj = compressed_projector_sum_rules(compress_mat, N, use_mkl=self._use_mkl)
        print_sp_matrix_size(proj, " P_(perm,trans,coset,sum):")
        tt6 = time.time()

        eigvecs = eigsh_projector_sumrule(proj)
        tt7 = time.time()
        print(" basis (size) =", eigvecs.shape)

        print("  t (spg_reps)            = ", tt00 - tt_begin)
        print("  t (init., perm)         = ", tt0 - tt00)
        print("  t (init., trans)        = ", tt1 - tt0)
        print("  t (dot, trans, perm)    = ", tt2 - tt1)
        print("  t (coset_reps_sum)      = ", tt3 - tt2)
        print("  t (dot, coset_reps_sum) = ", tt4 - tt3)
        print("  t (rot, trans, perm)    = ", tt5 - tt4)
        print("  t (proj_st)             = ", tt6 - tt5)
        print("  t (eigh(svd))           = ", tt7 - tt6)

        self._basis_set = eigvecs
        self._compression_matrix = compress_mat
        return self
