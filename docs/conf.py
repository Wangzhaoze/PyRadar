"""Sphinx configuration for the pyradar documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

project = "pyradar"
author = "pyradar contributors"
copyright = "2026, pyradar contributors"
release = "1.0.0rc1"
version = "1.0"

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
autodoc_typehints = "description"
numpydoc_show_class_members = False
myst_enable_extensions = ["colon_fence", "dollarmath", "fieldlist", "substitution"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "examples/data",
    "examples/output",
    "examples/GALLERY_HEADER.md",
    "tutorials/git-commands.md",
    "tutorials/pre-commit.md",
    "tutorials/pull-request.md",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "nested_sections": False,
    "ignore_pattern": (
        r"(?:^|[\\/])(?:data|configs)(?:[\\/]|$)"
        r"|(?:coloradar|radelft|rampcnn)_adc_to_pointcloud\.py$"
    ),
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
}

html_theme = "pydata_sphinx_theme"
html_title = f"pyradar {release}"
html_static_path: list[str] = []
html_theme_options = {
    "github_url": "https://github.com/Wangzhaoze/pyradar",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
