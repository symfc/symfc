# Symfc

Symfc is a Python-based force-constants solver that uses the supercell approach.
By employing an efficient projector-based algorithm that leverages crystal and
force-constant symmetries, it significantly reduces computational and memory
requirements. The code accepts displacement–force datasets as input and outputs
supercell force constants. Symfc supports the calculation of second-, third-,
and fourth-order force constants.

## Usage

Detailed documentation will be provided soon. In the meantime, please refer to
`api_symfc.py` for more information. Additionally, an [example
implementation](https://github.com/phonopy/phonopy/blob/master/phonopy/interface/symfc.py)
can be found in the phonopy code, particularly in the
`SymfcFCSolver._initialize` method.

## License

BSD-3-Clause.

## Citation of symfc

"Projector-based efficient estimation of force constants", A. Seko and A. Togo,
Phys. Rev. B, **110**, 214302 (2024)
[[doi](https://doi.org/10.1103/PhysRevB.110.214302)]
[[arxiv](https://arxiv.org/abs/2403.03588)].

```
@article{PhysRevB.110.214302,
  title = {Projector-based efficient estimation of force constants},
  author = {Seko, Atsuto and Togo, Atsushi},
  journal = {Phys. Rev. B},
  volume = {110},
  issue = {21},
  pages = {214302},
  numpages = {18},
  year = {2024},
  month = {Dec},
}
```

## Contributors

- {user}`Atsuto Seko <sekocha>` (Kyoto university)
- {user}`Atsushi Togo <atztogo>` (National Institute for Materials Science)

```{toctree}
:hidden:
install
changelog
```
