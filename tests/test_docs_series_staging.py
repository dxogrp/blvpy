from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage_docs import documentation_series, stage_documentation_series


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


def _manifest(root: Path) -> tuple[tuple[str, str, str | None], ...]:
    entries: list[tuple[str, str, str | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append(("directory", relative, None))
        else:
            entries.append(("file", relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(entries)


@pytest.mark.parametrize(
    ("package_version", "series"),
    [("0.3.0", "0.3"), ("0.3.1", "0.3"), ("10.20.30", "10.20")],
)
def test_documentation_series_uses_only_major_and_minor(package_version: str, series: str) -> None:
    assert documentation_series(package_version) == series


@pytest.mark.parametrize(
    "package_version",
    ["0.3", "v0.3.0", "0.03.0", "0.3.0rc1", "0.3.0.post1", "0.3.0+local", "1!0.3.0"],
)
def test_documentation_series_requires_a_canonical_stable_package_version(package_version: str) -> None:
    with pytest.raises(ValueError, match=r"canonical stable x\.y\.z"):
        documentation_series(package_version)


def test_first_series_populates_version_root_and_exact_switcher(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "series 0.3")
    site = tmp_path / "site"

    stage_documentation_series(build, site, "0.3.0", "https://docs.example.test/blvpy/")

    assert (site / "version" / "0.3" / "index.html").read_text(encoding="utf-8") == "series 0.3"
    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.3"
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


def test_patch_deployment_replaces_whole_series_and_latest_root(tmp_path: Path) -> None:
    site = tmp_path / "site"
    first = _make_build(tmp_path / "first", "package 0.3.0", obsolete=True)
    stage_documentation_series(first, site, "0.3.0", "https://docs.example.test")

    replacement = _make_build(tmp_path / "replacement", "package 0.3.1")
    stage_documentation_series(replacement, site, "0.3.1", "https://docs.example.test")

    series = site / "version" / "0.3"
    assert (series / "index.html").read_text(encoding="utf-8") == "package 0.3.1"
    assert (site / "index.html").read_text(encoding="utf-8") == "package 0.3.1"
    assert not (series / "obsolete.html").exists()
    assert not (series / "_static" / "obsolete.css").exists()
    assert not (site / "obsolete.html").exists()
    assert not (site / "_static" / "obsolete.css").exists()
    assert [entry["name"] for entry in _switcher(site)] == ["latest", "0.3"]


def test_older_series_deployment_does_not_replace_latest_root(tmp_path: Path) -> None:
    site = tmp_path / "site"
    latest = _make_build(tmp_path / "latest", "series 0.3")
    stage_documentation_series(latest, site, "0.3.0", "https://docs.example.test")
    latest_tree = _manifest(site / "version" / "0.3")

    older = _make_build(tmp_path / "older", "series 0.2")
    stage_documentation_series(older, site, "0.2.0", "https://docs.example.test")
    replacement = _make_build(tmp_path / "older-replacement", "series 0.2 refreshed")
    stage_documentation_series(replacement, site, "0.2.1", "https://docs.example.test")

    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.3"
    assert _manifest(site / "version" / "0.3") == latest_tree
    assert (site / "version" / "0.2" / "index.html").read_text(encoding="utf-8") == "series 0.2 refreshed"
    assert [entry["name"] for entry in _switcher(site)] == ["latest", "0.3", "0.2"]


def test_new_greatest_series_is_promoted_without_changing_history(tmp_path: Path) -> None:
    site = tmp_path / "site"
    old = _make_build(tmp_path / "old", "series 0.3")
    stage_documentation_series(old, site, "0.3.2", "https://docs.example.test")
    old_manifest = _manifest(site / "version" / "0.3")

    new = _make_build(tmp_path / "new", "series 0.4")
    stage_documentation_series(new, site, "0.4.0", "https://docs.example.test")

    assert (site / "index.html").read_text(encoding="utf-8") == "series 0.4"
    assert _manifest(site / "version" / "0.3") == old_manifest
    assert [entry["name"] for entry in _switcher(site)] == ["latest", "0.4", "0.3"]


def test_identical_series_rerun_is_idempotent(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "series 0.3")
    site = tmp_path / "site"
    stage_documentation_series(build, site, "0.3.0", "https://docs.example.test")
    before = _manifest(site)

    stage_documentation_series(build, site, "0.3.0", "https://docs.example.test")

    assert _manifest(site) == before


@pytest.mark.parametrize("legacy_name", ["0.3.0", "latest", "notes.txt"])
def test_mixed_or_unexpected_version_layout_is_rejected_before_mutation(tmp_path: Path, legacy_name: str) -> None:
    build = _make_build(tmp_path / "build", "new")
    site = tmp_path / "site"
    versions = site / "version"
    versions.mkdir(parents=True)
    legacy = versions / legacy_name
    if legacy_name.endswith(".txt"):
        legacy.write_text("unexpected", encoding="utf-8")
    else:
        legacy.mkdir()
        (legacy / "index.html").write_text("legacy", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected"):
        stage_documentation_series(build, site, "0.3.0", "https://docs.example.test")

    assert not (versions / "0.3").exists()
    assert legacy.exists()


def test_series_staging_preserves_git_metadata_and_rejects_unsafe_trees(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "safe")
    site = tmp_path / "site"
    git = site / ".git"
    git.mkdir(parents=True)
    marker = git / "HEAD"
    marker.write_text("ref: refs/heads/gh-pages", encoding="utf-8")
    stage_documentation_series(build, site, "0.3.0", "https://docs.example.test")
    assert marker.read_text(encoding="utf-8") == "ref: refs/heads/gh-pages"

    unsafe = _make_build(tmp_path / "unsafe", "unsafe")
    (unsafe / "link").symlink_to(unsafe / "index.html")
    before = _manifest(site)
    with pytest.raises(ValueError, match="symbolic link"):
        stage_documentation_series(unsafe, site, "0.3.1", "https://docs.example.test")
    assert _manifest(site) == before


@pytest.mark.parametrize("collision", ["version", "2.0"])
def test_series_staging_rejects_root_collisions(tmp_path: Path, collision: str) -> None:
    build = _make_build(tmp_path / "build", "unsafe")
    (build / collision).mkdir()

    with pytest.raises(ValueError, match="reserved site entry|conflicts with a version directory"):
        stage_documentation_series(build, tmp_path / "site", "0.3.0", "https://docs.example.test")

    assert not (tmp_path / "site").exists()


def test_series_staging_rejects_overlapping_paths(tmp_path: Path) -> None:
    build = _make_build(tmp_path / "build", "unsafe")

    with pytest.raises(ValueError, match="must not overlap"):
        stage_documentation_series(build, build / "site", "0.3.0", "https://docs.example.test")
