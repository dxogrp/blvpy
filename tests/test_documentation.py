from __future__ import annotations

import inspect
import json
import re
import runpy
import tomllib
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

import blvpy
from scripts.stage_docs import stage_documentation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
AUTODOC_PATTERN = re.compile(
    r"(?:```\{auto(?:class|function|exception|data|attribute)\}|\.\.\s+auto(?:class|function|exception|data|attribute)::)"
    r"\s+blvpy\.([A-Za-z_]\w*)",
    re.MULTILINE,
)
EXAMPLE_ROLE_PATTERN = re.compile(r"\{example\}`[^`]*<([^>]+)>`")


def _make_build(directory: Path, marker: str) -> Path:
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(marker, encoding="utf-8")
    assets = directory / "_static"
    assets.mkdir()
    (assets / "style.css").write_text(f"/* {marker} */", encoding="utf-8")
    return directory


def _switcher(site: Path) -> list[dict[str, object]]:
    return json.loads((site / "switcher.json").read_text(encoding="utf-8"))


def test_stage_first_documentation_release(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "release 0.1.0")
    site = tmp_path / "site"

    stage_documentation(build, site, "0.1.0", "https://dxogrp.github.io/blvpy/")

    assert (site / "version" / "0.1.0" / "index.html").read_text(encoding="utf-8") == "release 0.1.0"
    assert (site / "version" / "0.1.0" / "_static" / "style.css").is_file()
    assert (site / "index.html").read_text(encoding="utf-8") == "release 0.1.0"
    assert (site / "_static" / "style.css").read_text(encoding="utf-8") == "/* release 0.1.0 */"
    assert (site / ".nojekyll").is_file()
    assert _switcher(site) == [
        {
            "name": "latest",
            "version": "0.1.0",
            "url": "https://dxogrp.github.io/blvpy/",
            "preferred": True,
        },
        {
            "name": "0.1.0",
            "version": "0.1.0",
            "url": "https://dxogrp.github.io/blvpy/version/0.1.0/",
            "preferred": False,
        },
    ]
    assert 'http-equiv="refresh"' not in (site / "index.html").read_text(encoding="utf-8")


def test_stage_new_release_preserves_and_sorts_existing_versions(tmp_path: Path) -> None:
    site = tmp_path / "site"
    first = _make_build(tmp_path / "first", "original release")
    stage_documentation(first, site, "0.9.0", "https://docs.example.test/blvpy")
    original = (site / "version" / "0.9.0" / "index.html").read_bytes()
    (site / "obsolete.html").write_text("old root page", encoding="utf-8")
    obsolete_assets = site / "obsolete-assets"
    obsolete_assets.mkdir()
    (obsolete_assets / "old.css").write_text("old", encoding="utf-8")

    second = _make_build(tmp_path / "second", "new release")
    stage_documentation(second, site, "0.10.0", "https://docs.example.test/blvpy")

    assert (site / "version" / "0.9.0" / "index.html").read_bytes() == original
    assert (site / "index.html").read_text(encoding="utf-8") == "new release"
    assert not (site / "obsolete.html").exists()
    assert not obsolete_assets.exists()
    switcher = _switcher(site)
    assert [entry["name"] for entry in switcher] == ["latest", "0.10.0", "0.9.0"]
    assert [entry["version"] for entry in switcher] == ["0.10.0", "0.10.0", "0.9.0"]
    assert [entry["preferred"] for entry in switcher] == [True, False, False]
    assert [entry["url"] for entry in switcher] == [
        "https://docs.example.test/blvpy/",
        "https://docs.example.test/blvpy/version/0.10.0/",
        "https://docs.example.test/blvpy/version/0.9.0/",
    ]


def test_restaging_is_idempotent_but_rejects_changed_release(tmp_path: Path) -> None:
    site = tmp_path / "site"
    original = _make_build(tmp_path / "original", "original")
    stage_documentation(original, site, "0.1.0", "https://docs.example.test")
    stage_documentation(_make_build(tmp_path / "newer", "newer"), site, "0.2.0", "https://docs.example.test")
    stage_documentation(original, site, "0.1.0", "https://docs.example.test")
    root_before_rejected_change = (site / "index.html").read_bytes()

    changed = _make_build(tmp_path / "changed", "changed")
    with pytest.raises(ValueError, match="already published"):
        stage_documentation(changed, site, "0.1.0", "https://docs.example.test")

    assert (site / "version" / "0.1.0" / "index.html").read_text(encoding="utf-8") == "original"
    assert (site / "version" / "0.2.0" / "index.html").read_text(encoding="utf-8") == "newer"
    assert (site / "index.html").read_text(encoding="utf-8") == "newer"
    assert (site / "index.html").read_bytes() == root_before_rejected_change
    assert sum(bool(entry["preferred"]) for entry in _switcher(site)) == 1


def test_version_switcher_uses_exact_latest_label_and_contextual_selection() -> None:
    script = (DOCS_ROOT / "_static" / "version-switcher.js").read_text(encoding="utf-8")
    namespace = runpy.run_path(str(DOCS_ROOT / "conf.py"))

    assert namespace["html_js_files"] == [("version-switcher.js", {"defer": "defer"})]
    assert "option.textContent = label;" in script
    assert "(latest)" not in script
    assert "window.location.pathname.startsWith(numberedPath)" in script
    assert "entry.preferred === true" in script


@pytest.mark.parametrize("entry", ["version", "2.0"])
def test_stage_rejects_version_directory_collision(tmp_path: Path, entry: str) -> None:
    build = _make_build(tmp_path / "build", "release")
    (build / entry).mkdir()

    with pytest.raises(ValueError, match="reserved site entry|conflicts with a version directory"):
        stage_documentation(build, tmp_path / "site", "0.1.0", "https://docs.example.test")

    assert not (tmp_path / "site").exists()


def test_stage_rejects_symbolic_links_before_mutating_root(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "release")
    (build / "unsafe").symlink_to(build / "index.html")
    with pytest.raises(ValueError, match="symbolic links"):
        stage_documentation(build, tmp_path / "site", "0.1.0", "https://docs.example.test")

    safe_build = _make_build(tmp_path / "safe-build", "release")
    site = tmp_path / "existing-site"
    stage_documentation(safe_build, site, "0.1.0", "https://docs.example.test")
    (site / "unsafe").symlink_to(site / "index.html")
    with pytest.raises(ValueError, match="symbolic links"):
        stage_documentation(safe_build, site, "0.1.0", "https://docs.example.test")

    assert (site / "index.html").read_text(encoding="utf-8") == "release"
    assert (site / "unsafe").is_symlink()


@pytest.mark.parametrize("version", ["../0.1.0", "/0.1.0", "v0.1.0", "1.0-final", "not-a-version"])
def test_stage_rejects_invalid_or_unsafe_versions(tmp_path: Path, version: str) -> None:
    build = _make_build(tmp_path / "build", "release")

    with pytest.raises(ValueError, match="version"):
        stage_documentation(build, tmp_path / "site", version, "https://docs.example.test")

    assert not (tmp_path / "site").exists()


def test_stage_rejects_missing_build_and_overlapping_paths(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    with pytest.raises(ValueError, match="index.html"):
        stage_documentation(build, tmp_path / "site", "0.1.0", "https://docs.example.test")

    (build / "index.html").write_text("release", encoding="utf-8")
    with pytest.raises(ValueError, match="must not overlap"):
        stage_documentation(build, build / "site", "0.1.0", "https://docs.example.test")


def test_every_public_export_has_an_explicit_autodoc_entry() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    documented = set(AUTODOC_PATTERN.findall(documentation))

    assert documented == set(blvpy.__all__)


def test_markdown_sources_respect_configured_line_width() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    line_length = project["tool"]["ruff"]["line-length"]
    violations = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        if "_build" in path.parts:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if len(line) > line_length:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}:{len(line)}")

    assert violations == []


def test_gallery_links_are_release_pinned_and_target_every_example() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    examples = sorted(path.name for path in EXAMPLES_ROOT.glob("*.py"))
    linked_examples = sorted(EXAMPLE_ROLE_PATTERN.findall(documentation))

    namespace = runpy.run_path(str(DOCS_ROOT / "conf.py"))
    example_url, _ = namespace["extlinks"]["example"]

    assert examples
    assert linked_examples == examples
    assert example_url == (f"https://github.com/dxogrp/blvpy/blob/v{blvpy.__version__}/examples/%s")
    assert namespace["html_baseurl"] == f"https://dxogrp.github.io/blvpy/version/{blvpy.__version__}/"


def test_documented_public_signatures_match_release_contract() -> None:
    lower = inspect.signature(blvpy.LowerProblem)
    problem = inspect.signature(blvpy.BilevelProblem)
    solve = inspect.signature(blvpy.BilevelProblem.solve)
    diagnostics = inspect.signature(blvpy.BilevelProblem.gap_diagnostics)

    assert tuple(lower.parameters) == ("objective", "constraints", "parameters")
    assert lower.parameters["constraints"].default == ()
    assert lower.parameters["parameters"].default == ()
    assert tuple(problem.parameters) == ("upper_objective", "lower_problem", "upper_constraints")
    assert problem.parameters["upper_constraints"].default == ()
    assert tuple(solve.parameters) == (
        "self",
        "epsilon_initial",
        "epsilon_target",
        "contraction",
        "best_of",
        "feasibility_tolerance",
        "seed",
        "solver",
        "conic_solver",
        "solver_options",
        "conic_solver_options",
        "restoration",
        "max_retries",
        "verbose",
        "solver_verbose",
    )
    assert solve.parameters["epsilon_initial"].default == 1e-1
    assert solve.parameters["epsilon_target"].default == 1e-6
    assert solve.parameters["contraction"].default == 0.1
    assert solve.parameters["best_of"].default is None
    assert solve.parameters["feasibility_tolerance"].default == 1e-7
    assert solve.parameters["seed"].default is None
    assert solve.parameters["solver"].default == cp.IPOPT
    assert solve.parameters["conic_solver"].default == cp.CLARABEL
    assert solve.parameters["solver_options"].default is None
    assert solve.parameters["conic_solver_options"].default is None
    assert solve.parameters["restoration"].default is True
    assert solve.parameters["max_retries"].default == 8
    assert solve.parameters["verbose"].default is True
    assert solve.parameters["solver_verbose"].default is False
    assert tuple(diagnostics.parameters) == ("self", "result", "solver", "solver_options", "solver_verbose")
    assert diagnostics.parameters["solver"].default == cp.CLARABEL
    assert diagnostics.parameters["solver_options"].default is None
    assert diagnostics.parameters["solver_verbose"].default is False


def test_documentation_dependencies_and_make_targets_match_release_contract() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")

    assert project["project"]["version"] == blvpy.__version__
    assert project["project"]["requires-python"] == ">=3.12"
    assert "cvxpy>=1.9" in project["project"]["dependencies"]
    assert project["project"]["urls"]["Documentation"] == "https://dxogrp.github.io/blvpy/"
    assert "packaging>=25" in project["dependency-groups"]["dev"]
    assert set(project["dependency-groups"]["docs"]) == {
        "alabaster>=1,<2",
        "myst-parser>=5.1,<6",
        "sphinx>=9.1,<10",
    }
    for target in ("sync-docs", "docs", "check-docs", "check-examples", "test", "lint", "build"):
        assert re.search(rf"(?m)^\.PHONY: {re.escape(target)}$", makefile)


def test_documented_quickstart_and_gap_diagnostics_workflow() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = blvpy.LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = blvpy.BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
        upper_constraints=[x >= -1.0],
    )

    problem.validate()
    assert problem.is_dblp()
    result = problem.solve(verbose=False)
    diagnostics = problem.gap_diagnostics(result)

    assert result.succeeded, result.message
    assert np.isfinite(result.variable_values[x]).all()
    assert np.isfinite(result.variable_values[y]).all()
    assert diagnostics.source_gap is not None
    assert np.isfinite(diagnostics.source_gap)
