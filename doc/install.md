# Installation

PyPI and conda-forge packages are available.

- https://pypi.org/project/sympy/
- https://anaconda.org/conda-forge/symfc

## Requirement

- numpy
- scipy
- spglib

## Installation from source code

A simplest installation using conda-forge packages:

```bash
% conda create -n symfc -c conda-forge
% conda activate symfc
% conda install -c conda-forge numpy scipy spglib
% git clone https://github.com/symfc/symfc.git
% cd symfc
% pip install -e .
```
