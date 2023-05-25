"""Setup script."""
import numpy
from setuptools import Extension, setup

include_dirs = [numpy.get_include()]
sources = []
extra_compile_args = []
extra_link_args = []
define_macros = []

extension = Extension(
    "symfc._symfc",
    include_dirs=include_dirs,
    sources=["c/_symfc.c"] + sources,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)

setup(
    name="symfc",
    version="0.1",
    setup_requires=["numpy", "setuptools"],
    description="This is the symfc module.",
    author="Atsuto Seko",
    author_email="seko@cms.mtl.kyoto-u.ac.jp",
    install_requires=["numpy", "scipy", "phonopy", "spglib"],
    provides=["symfc"],
    platforms=["all"],
    ext_modules=[extension],
)
