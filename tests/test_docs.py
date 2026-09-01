from __future__ import annotations

import json
import re
import runpy
from pathlib import Path

import pytest

import blvpy
from scripts.stage_docs import documentation_series, stage_documentation_series

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
AUTODOC_PATTERN = re.compile(
    r"(?:```\{auto(?:class|function|exception|data|attribute)\}|\.\.\s+auto(?:class|function|exception|data|attribute)::)"
    r"\s+blvpy\.([A-Za-z_]\w*)",
    re.MULTILINE,
)
EXAMPLE_ROLE_PATTERN = re.compile(r"\{example\}`[^`]*<([^>]+)>`")


def _configuration(monkeypatch: pytest.MonkeyPatch, source_ref: str | None = None) -> dict[str, object]:
    if source_ref is None:
        monkeypatch.delenv("BLVPY_DOCS_SOURCE_REF", raising=False)
    else:
        monkeypatch.setenv("BLVPY_DOCS_SOURCE_REF", source_ref)
    return runpy.run_path(str(DOCS_ROOT / "conf.py"))


def _make_build(directory: Path, marker: str, *, obsolete: bool = False) -> Path:
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(marker, encoding="utf-8")
    assets = directory / "_static"
    assets.mkdir()
    (assets / "current.css").write_text(f"/* {marker} */", encoding="utf-8")
    if obsolete:
        (directory / "obsolete.html").write_text("obsolete", encoding="utf-8")
        (assets / "obsolete.css").write_text("obsolete", encoding="utf-8")
    return directory


def _switcher(site: Path) -> list[dict[str, object]]:
    return json.loads((site / "switcher.json").read_text(encoding="utf-8"))


def _snapshot(root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    entries: list[tuple[str, str, bytes | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append(("symlink", relative, str(path.readlink()).encode()))
        elif path.is_dir():
            entries.append(("directory", relative, None))
        else:
            entries.append(("file", relative, path.read_bytes()))
    return tuple(entries)


def test_every_public_export_has_an_explicit_autodoc_entry() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    documented = set(AUTODOC_PATTERN.findall(documentation))

    assert documented == set(blvpy.__all__)


def test_every_example_is_linked_from_the_gallery() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    examples = sorted(path.name for path in EXAMPLES_ROOT.glob("*.py"))
    linked_examples = sorted(EXAMPLE_ROLE_PATTERN.findall(documentation))

    assert examples
    assert linked_examples == examples


def test_local_documentation_uses_series_chrome_and_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = _configuration(monkeypatch)
    series = ".".join(blvpy.__version__.split(".")[:2])
    source_ref = f"v{blvpy.__version__}"

    assert configuration["package_version"] == blvpy.__version__
    assert configuration["documentation_series"] == series
    assert configuration["version"] == series
    assert configuration["release"] == series
    assert configuration["html_title"] == f"BLVPY {series}"
    assert configuration["html_baseurl"] == f"https://dxogrp.github.io/blvpy/version/{series}/"
    assert configuration["html_context"]["github_version"] == source_ref
    assert configuration["extlinks"]["example"][0] == (f"https://github.com/dxogrp/blvpy/blob/{source_ref}/examples/%s")


def test_deployed_documentation_uses_exact_source_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    source_ref = "0123456789abcdef0123456789abcdef01234567"
    configuration = _configuration(monkeypatch, source_ref)

    assert configuration["docs_source_ref"] == source_ref
    assert configuration["html_context"]["github_version"] == source_ref
    assert configuration["extlinks"]["example"][0] == (f"https://github.com/dxogrp/blvpy/blob/{source_ref}/examples/%s")


def test_documentation_series_accepts_stable_versions_and_rejects_other_forms() -> None:
    assert documentation_series("0.3.0") == "0.3"
    assert documentation_series("0.3.1") == "0.3"
    assert documentation_series("10.20.30") == "10.20"

    invalid = ["0.3", "v0.3.0", "0.03.0", "0.3.0rc1", "0.3.0+local", "1!0.3.0"]
    for package_version in invalid:
        with pytest.raises(ValueError, match=r"canonical stable x\.y\.z"):
            documentation_series(package_version)


def test_first_series_populates_root_and_exact_switcher(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "series 0.3")
    site = tmp_path / "site"
    git_head = site / ".git" / "HEAD"
    git_head.parent.mkdir(parents=True)
    git_head.write_text("ref: refs/heads/gh-pages", encoding="utf-8")

    stage_documentation_series(build, site, "0.3.0", "https://docs.example.test/blvpy/")

    assert (site / "version" / "0.3" / "index.html").read_text(encoding="utf-8") == "series 0.3"
    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.3"
    assert git_head.read_text(encoding="utf-8") == "ref: refs/heads/gh-pages"
    assert (site / ".nojekyll").is_file()
    assert _switcher(site) == [
        {
            "name": "latest",
            "version": "0.3",
            "url": "https://docs.example.test/blvpy/",
            "preferred": True,
        },
        {
            "name": "0.3",
            "version": "0.3",
            "url": "https://docs.example.test/blvpy/version/0.3/",
            "preferred": False,
        },
    ]


def test_same_series_replacement_removes_stale_files_and_is_idempotent(tmp_path: Path) -> None:
    site = tmp_path / "site"
    first = _make_build(tmp_path / "first", "package 0.3.0", obsolete=True)
    replacement = _make_build(tmp_path / "replacement", "package 0.3.1")
    stage_documentation_series(first, site, "0.3.0", "https://docs.example.test")

    stage_documentation_series(replacement, site, "0.3.1", "https://docs.example.test")

    series = site / "version" / "0.3"
    assert (series / "index.html").read_text(encoding="utf-8") == "package 0.3.1"
    assert (site / "index.html").read_text(encoding="utf-8") == "package 0.3.1"
    assert not (series / "obsolete.html").exists()
    assert not (series / "_static" / "obsolete.css").exists()
    assert not (site / "obsolete.html").exists()
    assert not (site / "_static" / "obsolete.css").exists()

    staged = _snapshot(site)
    stage_documentation_series(replacement, site, "0.3.1", "https://docs.example.test")
    assert _snapshot(site) == staged


def test_older_series_preserves_root_and_newer_series_is_promoted(tmp_path: Path) -> None:
    site = tmp_path / "site"
    current = _make_build(tmp_path / "current", "series 0.3")
    stage_documentation_series(current, site, "0.3.0", "https://docs.example.test")
    current_tree = _snapshot(site / "version" / "0.3")

    older = _make_build(tmp_path / "older", "series 0.2")
    stage_documentation_series(older, site, "0.2.1", "https://docs.example.test")
    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.3"
    assert _snapshot(site / "version" / "0.3") == current_tree

    newer = _make_build(tmp_path / "newer", "series 0.4")
    stage_documentation_series(newer, site, "0.4.0", "https://docs.example.test")
    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.4"
    assert _snapshot(site / "version" / "0.3") == current_tree
    assert [entry["name"] for entry in _switcher(site)] == ["latest", "0.4", "0.3", "0.2"]


def test_mixed_layout_and_symlink_are_rejected_without_mutation(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "series 0.3")
    site = tmp_path / "site"
    stage_documentation_series(build, site, "0.3.0", "https://docs.example.test")

    legacy = site / "version" / "0.3.0"
    legacy.mkdir()
    (legacy / "index.html").write_text("legacy", encoding="utf-8")
    before = _snapshot(site)
    with pytest.raises(ValueError, match="Unexpected documentation version directory"):
        stage_documentation_series(build, site, "0.3.1", "https://docs.example.test")
    assert _snapshot(site) == before

    (legacy / "index.html").unlink()
    legacy.rmdir()
    unsafe = site / "unsafe"
    unsafe.symlink_to("index.html")
    before = _snapshot(site)
    with pytest.raises(ValueError, match="symbolic link"):
        stage_documentation_series(build, site, "0.3.1", "https://docs.example.test")
    assert _snapshot(site) == before


def test_root_collisions_and_overlapping_paths_are_rejected(tmp_path: Path) -> None:
    for collision in ("version", "2.0"):
        build = _make_build(tmp_path / f"build-{collision}", "unsafe")
        (build / collision).mkdir()
        site = tmp_path / f"site-{collision}"
        with pytest.raises(ValueError, match="reserved site entry|conflicts with a version directory"):
            stage_documentation_series(build, site, "0.3.0", "https://docs.example.test")
        assert not site.exists()

    build = _make_build(tmp_path / "overlap-build", "unsafe")
    with pytest.raises(ValueError, match="must not overlap"):
        stage_documentation_series(build, build / "site", "0.3.0", "https://docs.example.test")
