(changelog)=

# Change Log

## Nov-30-2025: Version 1.6.0

- Sparse solver for FC2.
- Deprecated `displacements` and `forces` parameters for instantiating the
  `Symfc` class. Use its attributes after instantiation.

## Jul-21-2025: Version 1.5.4

- Improve eigensolver for large projector.
- Check symmetric property of projector and make it symmetric unless the
  symmetry is largely broken.

## Jul-15-2025: Version 1.5.3

- More optimization for basis set computations.

## Jul-14-2025: Version 1.5.2

- Improve memory efficiency and calculation performance for basis set
  computations using block matrix tree.

## Jul-10-2025: Version 1.5.1

- Improve memory efficiency and calculation performance for basis set
  computations. The current implementation shows particular effectiveness with
  `use_mkl=True` (requires sparse-dot-mkl) for large systems.

## Jul-3-2025: Version 1.5.0

- Improve memory efficiency in computing basis sets

## Jun-26-2025: Version 1.4.1

- Small fix for specific case

## May-31-2025: Version 1.4.0

- Maintenance release after refactoring

## Feb-25-2025: Version 1.3.4

- Maintenance release after refactoring

## Feb-11-2025: Version 1.3.3

- Fix minor translational invariance issues in O2, O3, and O4

## Feb-10-2025: Version 1.3.2

- Use numerically stable version of O3 translational invariance

## Feb-5-2025: Version 1.3.1

- Enabled cutoff parameter for fc2.

## Feb-4-2025: Version 1.3.0

- Add `Symfc.estimate_basis_set()`.
