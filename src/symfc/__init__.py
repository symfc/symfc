"""Force constants calculation code: Symfc."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .api_symfc import Symfc, eigh, eigsh

try:
    __version__ = _package_version("symfc")
except PackageNotFoundError:  # running from a source tree without an install
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.0.0"

__all__ = ["Symfc", "__version__", "eigh", "eigsh"]
