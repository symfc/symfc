"""Symmetry adapted basis sets of 3rd order force constants."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from scipy.sparse import csc_array, csr_array

from symfc.spg_reps import SpgRepsO3
from symfc.utils.eig_tools import (
    dot_product_sparse,
    eigsh_projector,
    eigsh_projector_sumrule,
)
from symfc.utils.matrix_tools_O3 import (
    compressed_projector_sum_rules,
    get_perm_compr_matrix_O3,
)
from symfc.utils.utils_O3 import (
    get_compr_coset_reps_sum_O3,
    get_lat_trans_compr_matrix_O3,
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
        super().__init__(supercell, log_level=log_level)
        self._spg_reps = SpgRepsO3(supercell)
        self._use_mkl = use_mkl

    @property
    def basis_set(self) -> Optional[csr_array]:
        """Return compressed basis set.

        n_c = len(compressed_indices).

        shape=(n_c*N*N*3*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_c,N,N,3,3,3).

        """
        return self._basis_set

    @property
    def compact_basis_set(self) -> Optional[csc_array]:
        """Return compressed basis set.

        shape=(n_a*N*N*3*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (n_a,N,N,3,3,3).

        """
        compress_mat = self.get_compr_mat_naNN333_or_NNN333(full_matrix=False)
        return dot_product_sparse(
            compress_mat.tocsr(), csc_array(self._basis_set), use_mkl=self._use_mkl
        )

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        """Return full (decompressed) basis set.

        shape=(N*N*N*3*3*3, n_bases), dtype='double'.

        Data in first dimension is ordered by (N,N,N,3,3,3).

        """
        if self._basis_set is None:
            return None
        compress_mat = self.get_compr_mat_naNN333_or_NNN333(full_matrix=True)
        return dot_product_sparse(csc_array(compress_mat), csc_array(self._basis_set))

    def run(self, use_mkl: bool = False) -> FCBasisSetO3:
        """Compute compressed force constants basis set.

        Parameters
        ----------
        use_mkl : bool


        """
        trans_perms = self._spg_reps.translation_permutations
        N = self._natom

        tt0 = time.time()
        c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
        tt1 = time.time()

        coset_reps_sum = get_compr_coset_reps_sum_O3(self._spg_reps)
        print_sp_matrix_size(coset_reps_sum, " R_(coset):")
        tt2 = time.time()

        c_pt = self._get_perm_trans_compr_matrix(c_trans, N)

        tt4 = time.time()

        compress_mat = self._get_total_compr_matrix(c_trans, c_pt, coset_reps_sum)
        tt5 = time.time()

        proj = compressed_projector_sum_rules(compress_mat, N, use_mkl=use_mkl)
        print_sp_matrix_size(proj, " P_(perm,trans,coset,sum):")
        tt6 = time.time()

        eigvecs = eigsh_projector_sumrule(proj)
        print(" basis (size) =", eigvecs.shape)

        tt7 = time.time()

        print("  t (init., trans)        = ", tt1 - tt0)
        print("  t (coset_reps_sum)      = ", tt2 - tt1)
        print("  t (dot, trans, perm)    = ", tt4 - tt2)
        print("  t (rot, trans, perm)    = ", tt5 - tt4)
        print("  t (proj_st)             = ", tt6 - tt5)
        print("  t (eigh(svd))           = ", tt7 - tt6)
        # print('  t (reconstruction)      = ', tt8-tt7)

        self._basis_set = eigvecs

        return self

    def get_compr_mat_naNN333_or_NNN333(self, full_matrix=False):
        """Regenerate compression matrix."""
        trans_perms = self._spg_reps.translation_permutations
        c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
        c_pt = self._get_perm_trans_compr_matrix(c_trans, self._natom)
        coset_reps_sum = get_compr_coset_reps_sum_O3(self._spg_reps)
        proj_rpt = dot_product_sparse(coset_reps_sum, c_pt)
        proj_rpt = dot_product_sparse(c_pt.T, proj_rpt)
        c_rpt = eigsh_projector(proj_rpt)
        compress_mat = dot_product_sparse(c_pt, c_rpt, self._use_mkl)
        if full_matrix:
            compress_mat = dot_product_sparse(
                c_trans, compress_mat, use_mkl=self._use_mkl
            )
        return compress_mat

    def _get_perm_trans_compr_matrix(self, c_trans: csr_array, N: int):
        """Return perm trans compression matrix.

        compression using C(pt)
            = eigvecs of C(trans).T @ C(perm) @ C(perm).T @ C(trans)
        proj_rpt = c_pt.T @ coset_reps_sum @ c_pt

        C(pt) = C(perm).T @ C(trans)

        """
        c_perm = get_perm_compr_matrix_O3(N)
        print_sp_matrix_size(c_perm, " C_perm:")
        c_pt = dot_product_sparse(c_perm.T, c_trans, use_mkl=self._use_mkl)
        print_sp_matrix_size(c_pt, " C_(perm,trans):")
        proj_pt = dot_product_sparse(c_pt.T, c_pt, use_mkl=self._use_mkl)
        print_sp_matrix_size(proj_pt, " P_(perm,trans):")
        c_pt = eigsh_projector(proj_pt)
        print_sp_matrix_size(c_pt, " C_(perm,trans,compressed):")
        return c_pt

    def _get_total_compr_matrix(
        self, c_trans: csr_array, c_pt: csr_array, coset_reps_sum: csr_array
    ) -> csr_array:
        """Return compression matrix.

        compress_mat = c_trans @ c_pt @ c_rpt

        c_trans : compression matrix by lattice translations
        c_pt : basis vectors of projection matrix of perm & trans
        c_rpt : basis vectors of projection matrix of  coset rots & perm & trans

        More precisely,

        [C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt] @ C_rpt
         = C_rpt @ [C_rpt.T @ C_pt @ C_trans.T] @ P_sum @ [C_trans @ C_pt @ C_rpt]
         = C_rpt @ compression_mat.T @ (I - P_sum^(c)) @ compression_mat
         = C_rpt @ compression_mat.T @ (I - Csum(c) @ Csum(c).T) @ compression_mat
         = C_rpt @ proj

        """
        proj_rpt = dot_product_sparse(coset_reps_sum, c_pt, use_mkl=self._use_mkl)
        proj_rpt = dot_product_sparse(c_pt.T, proj_rpt, use_mkl=self._use_mkl)
        print_sp_matrix_size(proj_rpt, " P_(perm,trans,coset):")
        c_rpt = eigsh_projector(proj_rpt)
        print_sp_matrix_size(c_rpt, " C_(perm,trans,coset):")
        compress_mat = dot_product_sparse(c_pt, c_rpt, use_mkl=self._use_mkl)
        compress_mat = dot_product_sparse(c_trans, compress_mat, use_mkl=self._use_mkl)
        print_sp_matrix_size(compress_mat, " compression matrix:")
        return compress_mat


def print_sp_matrix_size(c, header):
    """Show sparse matrix size."""
    print(header, c.shape, len(c.data))
