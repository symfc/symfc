"""Generate symmetrized force constants."""
from __future__ import annotations

import itertools
import time
from typing import Optional, Union

import numpy as np
import scipy
import spglib
from phonopy.structure.cells import compute_all_sg_permutations
from phonopy.utils import similarity_transformation
from scipy.sparse import coo_array, csr_array

from symfc.matrix_funcs import (
    convert_basis_sets_matrix_form,
    kron_c,
    to_serial,
    transform_n3n3_serial_to_nn33_serial,
)


class SymOpReps:
    """Matrix representations of symmetry operations."""

    def __init__(
        self,
        lattice: np.ndarray,
        positions: np.ndarray,
        numbers: np.ndarray,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        lattice : array_like
            Basis vectors as column vectors.
            shape=(3, 3), dtype='double'
        positions : array_like
            Position vectors given as column vectors.
            shape=(3, natom), dtype='double'
        numbers : array_like
            Atomic IDs idicated by integers larger or eqaul to 0.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._lattice = np.array(lattice, dtype="double", order="C")
        self._positions = np.array(positions, dtype="double", order="C")
        self._numbers = numbers
        self._log_level = log_level
        self._reps: Optional[list] = None

        self._run()

    @property
    def representations(self) -> Optional[list]:
        """Return matrix representations."""
        return self._reps

    def _run(self):
        rotations_inv, translations_inv = self._get_symops_inv()
        if self._log_level:
            print(" finding permutations ...")
        permutations_inv = compute_all_sg_permutations(
            self._positions.T, rotations_inv, translations_inv, self._lattice, 1e-5
        )
        if self._log_level:
            print(" setting representations (first order) ...")
        self._compute_reps(permutations_inv, rotations_inv)

    def _get_symops_inv(self, tol=1e-8) -> tuple[np.ndarray, np.ndarray]:
        """Return inverse symmetry operations.

        It is assumed that inverse symmetry operations are included in given
        symmetry operations up to lattice translation.

        Returns
        -------
        rotations_inv : array_like
            A set of rotation matrices of inverse space group operations.
            (n_symops, 3, 3), dtype='intc', order='C'
        translations_inv : array_like
            A set of translation vectors. It is assumed that inverse matrices are
            included in this set.
            (n_symops, 3), dtype='double'.

        """
        symops = spglib.get_symmetry(
            (self._lattice.T, self._positions.T, self._numbers)
        )
        rotations = symops["rotations"]
        translations = symops["translations"]
        rotations_inv = []
        translations_inv = []
        identity = np.eye(3, dtype=int)
        indices_found = [False] * len(rotations)
        for r, t in zip(rotations, translations):
            for i, (r_inv, t_inv) in enumerate(zip(rotations, translations)):
                if np.array_equal(r @ r_inv, identity):
                    diff = r_inv @ t + t_inv
                    diff -= np.rint(diff)
                    if np.linalg.norm(self._lattice @ np.abs(diff)) < tol:
                        rotations_inv.append(r_inv)
                        translations_inv.append(t_inv)
                        indices_found[i] = True
                        break
        assert len(rotations) == len(rotations_inv)
        assert len(translations) == len(translations_inv)
        assert all(indices_found)
        return (
            np.array(rotations_inv, dtype=rotations.dtype),
            np.array(translations_inv, dtype=translations.dtype),
        )

    def _compute_reps(self, permutations, rotations, tol=1e-10) -> None:
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

        """
        size = 3 * len(self._numbers)
        atom_indices = np.arange(len(self._numbers))  # [0, 1, 2, ..]
        self._reps = []
        for perm, r in zip(permutations, rotations):
            rot_cart = similarity_transformation(self._lattice, r)
            nonzero_r_row, nonzero_r_col = np.nonzero(np.abs(rot_cart) > tol)
            row = np.add.outer(perm * 3, nonzero_r_row).ravel()
            col = np.add.outer(atom_indices * 3, nonzero_r_col).ravel()
            nonzero_r_elems = [
                rot_cart[i, j] for i, j in zip(nonzero_r_row, nonzero_r_col)
            ]
            data = np.tile(nonzero_r_elems, len(self._numbers))

            # for atom1, atom2 in enumerate(perm):
            #    for i,j in zip(ids[0], ids[1]):
            #       id1 = 3 * atom2 + i
            #       id2 = 3 * atom1 + j
            #       row.append(id1)
            #       col.append(id2)
            #       data.append(rot[i,j])

            rep = coo_array((data, (row, col)), shape=(size, size))
            self._reps.append(rep)


class SymBasisSets:
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
            if np.all(np.abs(comm.data) < tol) is False:
                raise ValueError("Two projectors do not satisfy commutation rule.")

            U = proj_const.dot(nonzero_proj_R)
        else:
            if self._log_level:
                print("  - using approximate projection matrix ... ")

            proj_sum = _get_projector_sum_rule(self._natom)
            proj_perm = _get_projector_permutations(self._natom)

            # checking commutativity of two projectors
            comm = proj_R.dot(proj_sum) - proj_sum.dot(proj_R)
            if np.all(np.abs(comm.data) < tol) is False:
                raise ValueError("Two projectors do not satisfy " "commutation rule.")
            comm = proj_R.dot(proj_perm) - proj_perm.dot(proj_R)
            if np.all(np.abs(comm.data) < tol) is False:
                raise ValueError("Two projectors do not satisfy " "commutation rule.")
            comm = proj_sum.dot(proj_perm) - proj_perm.dot(proj_sum)
            if np.all(np.abs(comm.data) < tol) is False:
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


def get_projector_constraints(
    natom: int, with_permutation: bool = True, with_translation: bool = True
) -> csr_array:
    """Construct matrices of sum rule and permutation."""
    size_sq = (3 * natom) ** 2
    C = _get_projector_constraints_array(
        natom, with_permutation=with_permutation, with_translation=with_translation
    )
    Cinv = scipy.sparse.linalg.inv((C.T).dot(C))
    proj = scipy.sparse.eye(size_sq) - (C.dot(Cinv)).dot(C.T)
    return proj


def _get_projector_constraints_array(
    natom: int, with_permutation: bool = True, with_translation: bool = True
) -> csr_array:
    size_sq = (3 * natom) ** 2
    n, row, col, data = 0, [], [], []
    # sum rules
    if with_translation:
        n = _get_projector_constraints_sum_rule(row, col, data, natom, n)

    # permutation
    if with_permutation:
        n = _get_projector_constraints_permutations(row, col, data, natom, n)

    # Temporary fix
    # scipy.sparse.linalg.inv (finally splu) doesn't accept
    # "int_" (or list[int]) row and col values at scipy 1.11.1.
    dtype = "intc"
    row = np.array(row, dtype=dtype)
    col = np.array(col, dtype=dtype)
    return csr_array((data, (row, col)), shape=(size_sq, n))


def get_projector_constraints_sum_rule_array(natom: int) -> csr_array:
    """Construct matrices of sum rule only.

    Returns
    -------
    csr_array
        Each column contains N of 1 and others are zero.
        shape=((3N)**2, 9N)

    """
    size_sq = (3 * natom) ** 2
    n, row, col, data = 0, [], [], []
    n = _get_projector_constraints_sum_rule(row, col, data, natom, n)
    dtype = "intc"
    row = np.array(row, dtype=dtype)
    col = np.array(col, dtype=dtype)
    return csr_array((data, (row, col)), shape=(size_sq, n))


def _get_projector_constraints_sum_rule(
    row: list, col: list, data: list, natom: int, n: int
) -> int:
    for i in range(natom):
        for alpha, beta in itertools.product(range(3), range(3)):
            for j in range(natom):
                row.append(to_serial(i, alpha, j, beta, natom))
                col.append(n)
                data.append(1.0)
            n += 1
    return n


def get_projector_constraints_permutations_array(natom: int) -> csr_array:
    """Construct matrices of permutations only.

    Returns
    -------
    csr_array
        Each colum contains one 1 and one -1, and others are all zero.
        Diagonal elements in (NN33, NN33) representation are zero.
        shape=((3N)**2, 3N(3N-1))

    """
    size_sq = (3 * natom) ** 2
    n, row, col, data = 0, [], [], []
    n = _get_projector_constraints_permutations(row, col, data, natom, n)
    dtype = "intc"
    row = np.array(row, dtype=dtype)
    col = np.array(col, dtype=dtype)
    return csr_array((data, (row, col)), shape=(size_sq, n))


def _get_projector_constraints_permutations(
    row: list, col: list, data: list, natom: int, n: int
) -> int:
    for ia, jb in itertools.combinations(range(natom * 3), 2):
        i, a = ia // 3, ia % 3
        j, b = jb // 3, jb % 3
        id1 = to_serial(i, a, j, b, natom)
        id2 = to_serial(j, b, i, a, natom)
        row += [id1, id2]
        col += [n, n]
        data += [1, -1]
        n += 1
    return n


def _get_projector_sum_rule(natom) -> csr_array:
    size_sq = 9 * natom * natom
    C = _get_projector_sum_rule_array(natom)
    proj = scipy.sparse.eye(size_sq) - C
    return proj


def _get_projector_sum_rule_array(natom: int) -> csr_array:
    size_sq = 9 * natom * natom

    row, col, data = [], [], []
    block = np.tile(np.eye(9), (natom, natom))
    csr = csr_array(block)
    for i in range(natom):
        row1, col1 = csr.nonzero()
        row += (row1 + 9 * natom * i).tolist()
        col += (col1 + 9 * natom * i).tolist()
        data += (csr.data / natom).tolist()

    return csr_array((data, (row, col)), shape=(size_sq, size_sq))


def _get_projector_permutations(natom: int) -> csr_array:
    size = 3 * natom
    size_sq = size**2
    C = _get_projector_permutations_array(natom)
    proj = scipy.sparse.eye(size_sq) - C
    return proj


def _get_projector_permutations_array(natom: int) -> csr_array:
    size = 3 * natom
    size_sq = size**2

    row, col, data = [], [], []
    for ia, jb in itertools.combinations(range(size), 2):
        i, a = ia // 3, ia % 3
        j, b = jb // 3, jb % 3
        id1 = to_serial(i, a, j, b, natom)
        id2 = to_serial(j, b, i, a, natom)
        row += [id1, id2, id1, id2]
        col += [id1, id2, id2, id1]
        data += [0.5, 0.5, -0.5, -0.5]
    return csr_array((data, (row, col)), shape=(size_sq, size_sq))
