(changelog)=

# Change Log

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
