"""Symmetry adapted basis sets of 3rd order force constants."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from symfc.spg_reps import SpgRepsO3
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix_tools_O3 import (
    compressed_projector_sum_rules,
    permutation_symmetry_basis,
)
from symfc.utils.utils_O3 import (
    get_compr_coset_reps_sum_O3,
    get_lat_trans_compr_matrix_O3,
    get_lat_trans_decompr_indices_O3,
)

from .basis_sets_base import FCBasisSetBase


class FCBasisSetO3(FCBasisSetBase):
    """Symmetry adapted basis set for 3rd order force constants.

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
        self._spg_reps = SpgRepsO3(supercell)

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        """Return compressed basis set.

        shape=(n_a*N*N*3*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_a,N,N,3,3,3).

        ****************************
        Note this is a dummy method.
        ****************************

        """
        return self._basis_set

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        """Return full (decompressed) basis set.

        shape=(N*N*N*3*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (N,N,N,3,3,3).

        ****************************
        Note this is a dummy method.
        ****************************

        """
        if self._basis_set is None:
            return None
        return self._basis_set[self.decompression_indices, :]

    @property
    def decompression_indices(self) -> np.ndarray:
        """Return decompression indices in (N,N,N,3,3,3) order.

        ****************************
        Note this is a dummy method.
        ****************************

        """
        trans_perms = self.translation_permutations
        return get_lat_trans_decompr_indices_O3(trans_perms)

    @property
    def compression_indices(self) -> np.ndarray:
        """Return compression indices in (n_a,N,N,3,3,3) order.

        ****************************
        Note this is a dummy method.
        ****************************

        """
        return np.zeros(0)

    def run(self, use_mkl: bool = False, tol: float = 1e-8) -> FCBasisSetO3:
        """Compute compressed force constants basis set.

        Parameters
        ----------
        tol : float
            Tolerance to identify zero eigenvalues. Default=1e-8.

        """
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape

        tt00 = time.time()
        """C(permutation)"""
        c_perm = permutation_symmetry_basis(N)
        print_sp_matrix_size(c_perm, " C_perm:")
        tt0 = time.time()

        """C(trans)"""
        decompr_idx = get_lat_trans_decompr_indices_O3(trans_perms)
        c_trans = get_lat_trans_compr_matrix_O3(decompr_idx, N, n_lp)
        print_sp_matrix_size(c_trans, " C_trans:")
        tt1 = time.time()

        """C(pt) = C(perm).T @ C(trans)"""
        c_pt = dot_product_sparse(c_perm.transpose(), c_trans)
        print_sp_matrix_size(c_pt, " C_(perm,trans):")
        proj_pt = dot_product_sparse(c_pt.transpose(), c_pt)
        print_sp_matrix_size(proj_pt, " P_(perm,trans):")
        tt2 = time.time()

        coset_reps_sum = get_compr_coset_reps_sum_O3(self._spg_reps)
        print_sp_matrix_size(coset_reps_sum, " R_(coset):")
        tt3 = time.time()

        """
        compression using C(pt)
            = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
        proj_rpt = c_pt.transpose() @ coset_reps_sum @ c_pt
        """
        c_pt = eigsh_projector(proj_pt)
        print_sp_matrix_size(c_pt, " C_(perm,trans,compressed):")

        proj_rpt = dot_product_sparse(coset_reps_sum, c_pt)
        proj_rpt = dot_product_sparse(c_pt.transpose(), proj_rpt)
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
        compress_mat = dot_product_sparse(c_trans, c_pt)
        compress_mat = dot_product_sparse(compress_mat, c_rpt)
        print_sp_matrix_size(compress_mat, " compression matrix:")

        proj = compressed_projector_sum_rules(compress_mat, N, mkl=use_mkl)
        print_sp_matrix_size(proj, " P_(perm,trans,coset,sum):")
        tt6 = time.time()

        eigvecs = eigsh_projector_sumrule(proj)
        print(" basis (size) =", eigvecs.shape)

        tt7 = time.time()
        # full_eigvecs = dot_product_sparse(compress_mat, csr_matrix(eigvecs))
        # tt8 = time.time()

        print("  t (init., perm)         = ", tt0 - tt00)
        print("  t (init., trans)        = ", tt1 - tt0)
        print("  t (dot, trans, perm)    = ", tt2 - tt1)
        print("  t (coset_reps_sum)      = ", tt3 - tt2)
        print("  t (dot, coset_reps_sum) = ", tt4 - tt3)
        print("  t (rot, trans, perm)    = ", tt5 - tt4)
        print("  t (proj_st)             = ", tt6 - tt5)
        print("  t (eigh(svd))           = ", tt7 - tt6)
        # print('  t (reconstruction)      = ', tt8-tt7)

        self._basis_set = eigvecs

        return self


def print_sp_matrix_size(c, header):
    """Show sparse matrix size."""
    print(header, c.shape, len(c.data))
