from __future__ import annotations

import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

__version__: str = "unknown"
try:
    from yasfpy import __version__ as _yasfpy_version
except Exception:  # pragma: no cover
    pass
else:
    __version__ = _yasfpy_version

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------

project = "YASF"
author = "Mirza Arnaut"
copyright = f"2025, {author}"
version = release = __version__

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "autoapi.extension",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "**/.ipynb_checkpoints",
    # Stale AutoAPI output from removed modules.
    "autoapi/yasfpy/computers/**",
]

language = "en"

todo_include_todos = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -----------------------------------------------------------------------------
# MyST (Markdown)
# -----------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "tasklist",
]

# Add stable heading anchors for painless deep-linking.
myst_heading_anchors = 3

# -----------------------------------------------------------------------------
# Napoleon (NumPy-style docstrings)
# -----------------------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True
napoleon_use_ivar = True

# Keep the docs build clean when AutoAPI generates auxiliary pages.
# These warnings are expected in this setup and are non-actionable.
suppress_warnings = [
    "toc.not_included",
    "ref.duplicate",
    "autodoc.duplicate_object",
]

# -----------------------------------------------------------------------------
# autodoc
# -----------------------------------------------------------------------------

autoclass_content = "class"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}

# sphinx-autodoc-typehints
always_document_param_types = True
typehints_fully_qualified = False
typehints_document_rtype = True
always_use_bars_union = True

# -----------------------------------------------------------------------------
# AutoAPI
# -----------------------------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(ROOT / "yasfpy")]
autoapi_root = "autoapi"

# The Streamlit apps are for interactive exploration and pull in heavy optional
# dependencies; keep them out of the API reference.
autoapi_ignore = [
    "*migrations*",
    "apps/*",
    "apps/**",
    "*/apps/*",
    "*/apps/**",
]

# Keep the API docs linked from api.md, not injected into index.
autoapi_add_toctree_entry = False

# AutoAPI generates `.rst` into `autoapi_root` during builds.
# Keep them on disk so Sphinx can copy sources into `_sources/` reliably.
# These files are ignored by git (see `.gitignore`).
autoapi_keep_files = True

# One page per module, including members.
autoapi_own_page_level = "module"
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
    "show-inheritance",
]

# -----------------------------------------------------------------------------
# BibTeX
# -----------------------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -----------------------------------------------------------------------------
# Intersphinx
# -----------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/darkmode-image.css",
]

# Avoid embedding the release/version in the navbar title.
html_title = "YASF documentation"

html_theme_options = {
    "logo": {
        "text": "YASF",
        "alt_text": "YASF documentation - Home",
        "image_light": "_static/logo_black.svg",
        "image_dark": "_static/logo_white.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/AGBV/YASF",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/yasfpy/",
            "icon": "fa-brands fa-python",
        },
    ],
    "search_bar_text": "Search the docs...",
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "AGBV",
    "github_repo": "YASF",
    "github_version": "main",
    "doc_path": "docs/source",
}
