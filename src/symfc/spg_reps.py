"""Reps of space group operations with respect to atomic coordinate basis."""
from __future__ import annotations

from typing import Optional

import numpy as np
import spglib
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import compute_all_sg_permutations
from phonopy.utils import similarity_transformation
from scipy.sparse import coo_array

import symfc._symfc as symfcc


class SpgReps:
    """Reps of space group operations with respect to atomic coordinate basis."""

    def __init__(self, supercell: PhonopyAtoms):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.

        """
        self._lattice = np.array(supercell.cell, dtype="double", order="C")
        self._positions = np.array(
            supercell.scaled_positions, dtype="double", order="C"
        )
        self._numbers = supercell.numbers
        self._reps: Optional[list[coo_array]] = None
        self._r2_reps: Optional[list[coo_array]] = None
        self._unique_rotation_indices: Optional[np.ndarray] = None
        self._translation_permutations: Optional[np.ndarray] = None

        self._prepare()

    @property
    def numbers(self):
        """Return atomic numbers."""
        return self._numbers

    @property
    def representations(self) -> Optional[list[coo_array]]:
        """Return 3Nx3N matrix representations."""
        return self._reps

    @property
    def translation_permutations(self) -> Optional[np.ndarray]:
        """Return permutations by lattice translation.

        Returns
        --------
        Atom indices after lattice translations.
        shape=(lattice_translations, supercell_atoms), dtype=int

        """
        return self._translation_permutations

    @property
    def unique_rotation_indices(self) -> Optional[np.ndarray]:
        """Return indices of coset representatives of space group operations."""
        return self._unique_rotation_indices

    @property
    def r2_reps(self) -> Optional[list[coo_array]]:
        """Return 2nd rank tensor rotation matricies."""
        return self._r2_reps

    def get_fc2_rep(self, i: int, mode: str = "new") -> Optional[coo_array]:
        """Return i'th matrix representation for fc2."""
        if mode == "new":
            return self._get_fc2_rep(i)
        else:
            if self._reps is None:
                self._reps = self._compute_reps_order1(
                    self._permutations, self._rotations, self._unique_rotation_indices
                )
                self._check_rep_dtypes()
            return self._get_fc2_rep_by_kron(i)

    def get_sigma2_rep(self, i: int) -> coo_array:
        """Compute and return i-th atomic pair permutation matrix.

        Parameters
        ----------
        i : int
            Index of coset presentations of space group operations.

        """
        data, row, col, shape = self._get_sigma2_rep_data(i)
        return coo_array((data, (row, col)), shape=shape)

    def _compute_r2_reps(self, tol=1e-10) -> list:
        """Compute and return 2nd rank tensor rotation matricies."""
        uri = self._unique_rotation_indices
        r2_reps = []
        for r in self._rotations[uri]:
            r_c = self._lattice.T @ r @ np.linalg.inv(self._lattice.T)
            r2_rep = np.kron(r_c, r_c)
            row, col = np.nonzero(np.abs(r2_rep) > tol)
            data = r2_rep[(row, col)]
            r2_reps.append(coo_array((data, (row, col)), shape=r2_rep.shape))
        return r2_reps

    def _get_fc2_rep_by_kron(self, i: int) -> Optional[coo_array]:
        """Return i'th matrix representation for fc2.

        kron
        ----
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

        p = 3*N, R=(r,s) and R=(v,w) in (3N, 3N).
        i = r // 3
        j = r % 3
        k = s // 3
        l = s % 3
        I = v // 3
        J = v % 3
        K = w // 3
        L = w % 3

        kron(R, R)_(pr+v, ps+w) = R(r,s)*R(v,w)  (3N*3N, 3N*3N)
        kron(R, R)_(r, v, s, w) = R(r,s)*R(v,w)  (3N,3N, 3N,3N)
        kron(R, R)_(i, j, I, J, k, l, K, L) = R(r,s)*R(v,w)  (N,3,N,3, N,3,N,3)
        kron(R, R)_(i, I, j, J, k, K, l, L) = R(r,s)*R(v,w)  (N,N,3,3, N,N,3,3)
        kron(R, R)_(i*9N+I*9+j*3+J, k*9N+K*9+l*3+L) = R(r,s)*R(v,w)  (N,N,3,3, N,N,3,3)

        Note
        ----
        At some version of scipy, dtype of coo_array.col and coo_array.row changed.
        Here the dtype is assumed 'intc' (<1.11) or 'int_' (>=1.11).

        """
        if self._reps is None:
            return None
        row_dtype = self._reps[0].row.dtype
        col_dtype = self._reps[0].col.dtype
        data_dtype = self._reps[0].data.dtype
        natom = len(self._numbers)
        rmat = self._reps[i]
        size_sq = (3 * natom) ** 2
        size = rmat.row.shape[0] ** 2
        row = np.zeros(size, dtype=row_dtype)
        col = np.zeros(size, dtype=col_dtype)
        data = np.zeros(size, dtype=data_dtype)
        args = (row, col, data, rmat.row, rmat.col, rmat.data, 3 * natom)
        if col_dtype is np.dtype("intc") and row_dtype is np.dtype("intc"):
            symfcc.kron_nn33_int(*args)
        elif col_dtype is np.dtype("int_") and row_dtype is np.dtype("int_"):
            symfcc.kron_nn33_long(*args)
        else:
            raise RuntimeError("Incompatible data type of rows and cols of coo_array.")
        return coo_array((data, (row, col)), shape=(size_sq, size_sq), dtype="double")

    def _prepare(self):
        self._rotations, translations = self._get_symops()
        self._permutations = compute_all_sg_permutations(
            self._positions, self._rotations, translations, self._lattice.T, 1e-5
        )
        self._translation_permutations = self._get_translation_permutations(
            self._permutations, self._rotations
        )
        self._unique_rotation_indices = self._get_unique_rotation_indices(
            self._rotations
        )
        N = len(self._numbers)
        a = np.arange(N)
        self._atom_pairs = np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2)
        self._coeff = np.array([1, N], dtype=int)
        self._col = self._atom_pairs @ self._coeff
        self._data = np.ones(N * N, dtype=int)
        self._r2_reps = self._compute_r2_reps()

    def _check_rep_dtypes(self):
        """Return data types of reps."""
        if self._reps is None:
            return None
        row_dtype = self._reps[0].row.dtype
        col_dtype = self._reps[0].col.dtype
        data_dtype = self._reps[0].data.dtype
        assert row_dtype in (np.dtype("intc"), np.dtype("int_"))
        assert self._reps[0].row.flags.contiguous
        assert col_dtype in (np.dtype("intc"), np.dtype("int_"))
        assert self._reps[0].col.flags.contiguous
        assert data_dtype is np.dtype("double")
        assert self._reps[0].data.flags.contiguous

    def _get_translation_permutations(self, permutations, rotations) -> np.ndarray:
        eye3 = np.eye(3, dtype=int)
        trans_perms = []
        for r, perm in zip(rotations, permutations):
            if np.array_equal(r, eye3):
                trans_perms.append(perm)
        return np.array(trans_perms, dtype="intc", order="C")

    def _get_unique_rotation_indices(self, rotations: np.ndarray) -> list[int]:
        unique_rotations: list[np.ndarray] = []
        indices = []
        for i, r in enumerate(rotations):
            is_found = False
            for ur in unique_rotations:
                if np.array_equal(r, ur):
                    is_found = True
                    break
            if not is_found:
                unique_rotations.append(r)
                indices.append(i)
        return indices

    def _get_symops(self) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetry operations.

        The set of inverse operations is the same as the set of the operations.

        Returns
        -------
        rotations : array_like
            A set of rotation matrices of inverse space group operations.
            (n_symops, 3, 3), dtype='intc', order='C'
        translations : array_like
            A set of translation vectors. It is assumed that inverse matrices are
            included in this set.
            (n_symops, 3), dtype='double'.

        """
        symops = spglib.get_symmetry((self._lattice, self._positions, self._numbers))
        return symops["rotations"], symops["translations"]

    def _compute_reps_order1(
        self,
        permutations: np.ndarray,
        rotations: np.ndarray,
        rotation_indices: Optional[list[int]] = None,
        tol=1e-10,
    ) -> list[coo_array]:
        """Construct representation matrices of rotations.

        Permutation of atoms by r, perm(r) = [0 1], means the permutation matrix:
            [1 0]
            [0 1]
        Rotation matrix in Cartesian coordinates:
            [0 1 0]
        r = [1 0 0]
            [0 0 1]

        Its representation matrix of perm(r) and r becomes

        [0 1 0 0 0 0]
        [1 0 0 0 0 0]
        [0 0 1 0 0 0]
        [0 0 0 0 1 0]
        [0 0 0 1 0 0]
        [0 0 0 0 0 1]

        Note
        ----
        np.add.outer(a, b).ravel() = [i + j for i in a for j in b].

        """
        size = 3 * len(self._numbers)
        atom_indices = np.arange(len(self._numbers))
        if rotation_indices is None:
            idxs = list(range(len(self._rotations)))
        else:
            idxs = rotation_indices
        reps = []
        for perm, r in zip(permutations[idxs], rotations[idxs]):
            rot_cart = similarity_transformation(self._lattice.T, r)
            nonzero_r_row, nonzero_r_col = np.nonzero(np.abs(rot_cart) > tol)
            row = np.add.outer(perm * 3, nonzero_r_row).ravel()
            col = np.add.outer(atom_indices * 3, nonzero_r_col).ravel()
            nonzero_r_elems = [
                rot_cart[i, j] for i, j in zip(nonzero_r_row, nonzero_r_col)
            ]
            data = np.tile(nonzero_r_elems, len(self._numbers))
            reps.append(coo_array((data, (row, col)), shape=(size, size)))
        return reps

    def _get_sigma2_rep_data(self, i: int) -> coo_array:
        uri = self._unique_rotation_indices
        permutation = self._permutations[uri[i]]
        NN = len(self._numbers) ** 2
        row = permutation[self._atom_pairs] @ self._coeff
        return self._data, row, self._col, (NN, NN)

    def _get_fc2_rep(self, i: int):
        """Return fc2 rep.

        Slightly faster than `scipy.sparse.kron(_get_sigma2_rep(permutation),
        r2_rep)`, but not fast enough.

        """
        r2_rep = self._r2_reps[i]
        data, row, col, shape = self._get_sigma2_rep_data(i)
        NN9 = shape[0] * 9
        n = r2_rep.nnz
        row = (np.repeat(row * 9, n).reshape(-1, n) + r2_rep.row).ravel()
        col = (np.repeat(col * 9, n).reshape(-1, n) + r2_rep.col).ravel()
        data = np.kron(data, r2_rep.data)
        return coo_array((data, (row, col)), shape=(NN9, NN9), dtype="double")
