# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "symfc"
copyright = "2024, symfc project"
author = "Atsuto Seko"

version = "1.6"
release = "1.6.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.mathjax",
    "myst_parser",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.extlinks",
]
myst_enable_extensions = ["linkify", "dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "Symfc v.%s" % release

extlinks = {
    "issue": ("https://github.com/symfc/symfc/issues/%s", "issue %s"),
    "path": ("https://github.com/symfc/symfc/tree/develop/%s", "%s"),
    "user": ("https://github.com/%s", "%s"),
}
