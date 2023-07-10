"""Reps of space group operations with respect to atomic coordinate basis."""
from typing import Optional

import numpy as np
import spglib
from phonopy.structure.cells import compute_all_sg_permutations
from phonopy.utils import similarity_transformation
from scipy.sparse import coo_array


class SymOpReps:
    """Reps of space group operations with respect to atomic coordinate basis."""

    def __init__(
        self,
        lattice: np.ndarray,
        positions: np.ndarray,
        numbers: np.ndarray,
        pure_translation_only: bool = False,
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
        self._pure_translation_only = pure_translation_only
        self._log_level = log_level
        self._reps: Optional[list] = None

        self._run()

    @property
    def representations(self) -> Optional[list]:
        """Return matrix representations."""
        return self._reps

    def _run(self):
        rotations_inv, translations_inv = self._get_symops_inv()
        if self._pure_translation_only:
            identity = np.eye(3, dtype=int)
            _rots = []
            _trans = []
            for r, t in zip(rotations_inv, translations_inv):
                if (r == identity).all():
                    _rots.append(r)
                    _trans.append(t)
            rotations_inv = np.array(_rots, dtype=rotations_inv.dtype)
            translations_inv = np.array(_trans, dtype=translations_inv.dtype)

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
