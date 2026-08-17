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

    assert (site / "0.1.0" / "index.html").read_text(encoding="utf-8") == "release 0.1.0"
    assert (site / "0.1.0" / "_static" / "style.css").is_file()
    assert (site / ".nojekyll").is_file()
    assert _switcher(site) == [
        {
            "name": "0.1.0",
            "version": "0.1.0",
            "url": "https://dxogrp.github.io/blvpy/0.1.0/",
            "preferred": True,
        }
    ]
    redirect = (site / "index.html").read_text(encoding="utf-8")
    assert "url=https://dxogrp.github.io/blvpy/0.1.0/" in redirect


def test_stage_new_release_preserves_and_sorts_existing_versions(tmp_path: Path) -> None:
    site = tmp_path / "site"
    first = _make_build(tmp_path / "first", "original release")
    stage_documentation(first, site, "0.9.0", "https://docs.example.test/blvpy")
    original = (site / "0.9.0" / "index.html").read_bytes()

    second = _make_build(tmp_path / "second", "new release")
    stage_documentation(second, site, "0.10.0", "https://docs.example.test/blvpy")

    assert (site / "0.9.0" / "index.html").read_bytes() == original
    assert [entry["version"] for entry in _switcher(site)] == ["0.10.0", "0.9.0"]
    assert [entry["preferred"] for entry in _switcher(site)] == [True, False]
    assert "0.10.0/" in (site / "index.html").read_text(encoding="utf-8")


def test_restaging_is_idempotent_but_rejects_changed_release(tmp_path: Path) -> None:
    site = tmp_path / "site"
    original = _make_build(tmp_path / "original", "original")
    stage_documentation(original, site, "0.1.0", "https://docs.example.test")
    stage_documentation(_make_build(tmp_path / "newer", "newer"), site, "0.2.0", "https://docs.example.test")
    stage_documentation(original, site, "0.1.0", "https://docs.example.test")

    changed = _make_build(tmp_path / "changed", "changed")
    with pytest.raises(ValueError, match="already published"):
        stage_documentation(changed, site, "0.1.0", "https://docs.example.test")

    assert (site / "0.1.0" / "index.html").read_text(encoding="utf-8") == "original"
    assert (site / "0.2.0" / "index.html").read_text(encoding="utf-8") == "newer"
    assert sum(bool(entry["preferred"]) for entry in _switcher(site)) == 1


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

    assert len(examples) == 8
    assert linked_examples == examples
    assert example_url == (f"https://github.com/dxogrp/blvpy/blob/v{blvpy.__version__}/examples/%s")


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
    assert problem.is_dbp()
    result = problem.solve(verbose=False)
    diagnostics = problem.gap_diagnostics(result)

    assert result.succeeded, result.message
    assert np.isfinite(result.variable_values[x]).all()
    assert np.isfinite(result.variable_values[y]).all()
    assert diagnostics.source_gap is not None
    assert np.isfinite(diagnostics.source_gap)
