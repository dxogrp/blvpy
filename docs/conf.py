"""Sphinx configuration for the BLVPY documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blvpy import __version__  # noqa: E402

project = "BLVPY"
author = "Hao Zhu"
copyright = "2026, Hao Zhu and BLVPY contributors"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Release-specific links are derived from the package version so a release
# documentation build always points to examples from its matching Git tag.
extlinks = {
    "example": (
        f"https://github.com/dxogrp/blvpy/blob/v{release}/examples/%s",
        "%s",
    )
}

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_typehints_format = "short"
autosummary_generate = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = True

# Keep the extension enabled without making strict local builds depend on
# downloading third-party inventories. External projects are linked directly.
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}
nitpicky = True
nitpick_ignore_regex = [
    ("py:class", r"(ArrayLike|NDArray|Mapping|Sequence|Generator|Expression|Objective|Constraint)"),
    ("py:class", r"(collections|cvxpy|numpy|numpy\.typing|scipy|types|typing)\..*"),
]

html_theme = "alabaster"
html_title = f"BLVPY {release}"
html_baseurl = f"https://dxogrp.github.io/blvpy/{release}/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["version-switcher.js"]
html_theme_options = {
    "description": "Disciplined bilevel programming in Python",
    "fixed_sidebar": True,
    "github_button": True,
    "github_repo": "blvpy",
    "github_type": "star",
    "github_user": "dxogrp",
    "page_width": "1120px",
    "show_powered_by": False,
    "sidebar_width": "270px",
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
        "versions.html",
    ]
}
html_context = {
    "docs_switcher_url": "https://dxogrp.github.io/blvpy/switcher.json",
    "github_version": f"v{release}",
}


def _public_signature(app, what, name, obj, options, signature, return_annotation):
    """Hide private dataclass constructor fields from the public reference."""
    del app, what, obj, options
    if name == "blvpy.CanonicalLowerProblem":
        return "", return_annotation
    return signature, return_annotation


def setup(app):
    """Register documentation-only rendering hooks."""
    app.connect("autodoc-process-signature", _public_signature)
