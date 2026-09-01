"""Publish mutable BLVPY documentation series on GitHub Pages."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from packaging.version import InvalidVersion, Version

VERSION_DIRECTORY = "version"
PACKAGE_VERSION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
SERIES_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")


def _canonical_version(value: str) -> tuple[str, Version]:
    """Return a safe canonical PEP 440 version and its parsed representation."""
    try:
        parsed = Version(value)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid documentation version: {value!r}.") from exc

    canonical = str(parsed)
    if value != canonical or not canonical or canonical in {".", ".."}:
        raise ValueError(f"Documentation version must be canonical PEP 440: {value!r}.")
    if Path(canonical).name != canonical or "/" in canonical or "\\" in canonical:
        raise ValueError(f"Unsafe documentation version path: {value!r}.")
    return canonical, parsed


def _canonical_package_version(value: str) -> Version:
    """Return a canonical stable three-component package version."""
    if PACKAGE_VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Package version must be a canonical stable x.y.z version: {value!r}.")
    canonical, parsed = _canonical_version(value)
    if canonical != value or len(parsed.release) != 3:
        raise ValueError(f"Package version must be a canonical stable x.y.z version: {value!r}.")
    return parsed


def _canonical_series(value: str) -> tuple[str, tuple[int, int]]:
    """Return a canonical major/minor documentation series."""
    if SERIES_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Documentation series must be a canonical x.y version: {value!r}.")
    canonical, parsed = _canonical_version(value)
    if canonical != value or len(parsed.release) != 2:
        raise ValueError(f"Documentation series must be a canonical x.y version: {value!r}.")
    return canonical, (parsed.major, parsed.minor)


def documentation_series(package_version: str) -> str:
    """Derive the mutable documentation series for a stable package version."""
    parsed = _canonical_package_version(package_version)
    return f"{parsed.major}.{parsed.minor}"


def _canonical_base_url(value: str) -> str:
    """Validate and normalize the public documentation base URL."""
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("The documentation base URL must be an absolute HTTP(S) URL.")
    if parts.query or parts.fragment:
        raise ValueError("The documentation base URL cannot contain a query or fragment.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _validate_tree(root: Path) -> None:
    """Reject symbolic links and unsupported filesystem entries in a tree."""
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Documentation trees cannot contain symbolic links: {path}.")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"Unsupported documentation filesystem entry: {path}.")


def _validate_root_tree(root: Path) -> None:
    """Validate a documentation tree before copying it to the site root."""
    _validate_tree(root)
    for path in root.iterdir():
        if path.name in {".git", VERSION_DIRECTORY}:
            raise ValueError(f"Documentation tree contains a reserved site entry: {path}.")
        try:
            _canonical_version(path.name)
        except ValueError:
            continue
        raise ValueError(f"Documentation root entry conflicts with a version directory: {path}.")


def _replace_series_tree(source: Path, target: Path) -> None:
    """Replace one mutable series tree with a complete validated build."""
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(f"Documentation series target is not a safe directory: {target}.")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        _validate_tree(temporary)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _version_root(site_dir: Path) -> Path:
    root = site_dir / VERSION_DIRECTORY
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"Documentation version root is not a safe directory: {root}.")
    return root


def _published_series(site_dir: Path) -> list[tuple[str, tuple[int, int]]]:
    """Return every published series, rejecting mixed or unsafe layouts."""
    root = _version_root(site_dir)
    if not root.exists():
        return []

    versions: list[tuple[str, tuple[int, int]]] = []
    for path in root.iterdir():
        if path.is_symlink():
            raise ValueError(f"Documentation trees cannot contain symbolic links: {path}.")
        if not path.is_dir():
            raise ValueError(f"Unexpected entry in the documentation version directory: {path}.")
        try:
            canonical, parsed = _canonical_series(path.name)
        except ValueError as exc:
            raise ValueError(f"Unexpected documentation version directory: {path}.") from exc
        _validate_root_tree(path)
        versions.append((canonical, parsed))
    return sorted(versions, key=lambda item: item[1], reverse=True)


def _write_text_atomic(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_site_metadata(
    site_dir: Path,
    base_url: str,
    versions: list[tuple[str, tuple[int, int]]],
) -> None:
    if not versions:
        raise ValueError("The staged site does not contain a released documentation version.")

    latest_version = versions[0][0]
    entries = [
        {
            "name": "latest",
            "version": latest_version,
            "url": f"{base_url}/",
            "preferred": True,
        }
    ]
    entries.extend(
        {
            "name": version,
            "version": version,
            "url": f"{base_url}/{VERSION_DIRECTORY}/{version}/",
            "preferred": False,
        }
        for version, _ in versions
    )
    _write_text_atomic(site_dir / "switcher.json", json.dumps(entries, indent=2) + "\n")
    _write_text_atomic(site_dir / ".nojekyll", "")


def _remove_root_documentation(site_dir: Path) -> None:
    """Remove root documentation while preserving series and Git metadata."""
    preserved_names = {VERSION_DIRECTORY, ".git"}
    root_entries = [path for path in site_dir.iterdir() if path.name not in preserved_names]
    for path in root_entries:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _copy_tree_contents(source: Path, destination: Path) -> None:
    """Copy the contents of one validated documentation tree."""
    for path in source.iterdir():
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def _refresh_documentation_root(
    site_dir: Path,
    base_url: str,
    versions: list[tuple[str, tuple[int, int]]],
) -> None:
    """Serve the greatest published series directly from the Pages root."""
    if not versions:
        raise ValueError("The staged site does not contain a released documentation version.")

    version_root = _version_root(site_dir)
    preferred = version_root / versions[0][0]
    _remove_root_documentation(site_dir)
    _copy_tree_contents(preferred, site_dir)
    _write_site_metadata(site_dir, base_url, versions)


def _validate_existing_site(site_dir: Path) -> list[tuple[str, tuple[int, int]]]:
    """Validate mutable site content before changing a series or metadata."""
    if site_dir.is_symlink() or (site_dir.exists() and not site_dir.is_dir()):
        raise ValueError(f"Documentation site is not a safe directory: {site_dir}.")
    if not site_dir.exists():
        return []

    git_metadata = site_dir / ".git"
    if git_metadata.is_symlink():
        raise ValueError(f"Documentation sites cannot contain symbolic links: {git_metadata}.")
    versions = _published_series(site_dir)
    for path in site_dir.iterdir():
        if path.name in {VERSION_DIRECTORY, ".git"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Documentation sites cannot contain symbolic links: {path}.")
        if path.is_dir():
            _validate_tree(path)
        elif not path.is_file():
            raise ValueError(f"Unsupported documentation filesystem entry: {path}.")
    return versions


def stage_documentation_series(build_dir: Path, site_dir: Path, package_version: str, base_url: str) -> None:
    """Replace a mutable documentation series and refresh root when it is latest."""
    series = documentation_series(package_version)
    normalized_url = _canonical_base_url(base_url)
    source_input = build_dir.expanduser()
    destination_input = site_dir.expanduser()
    if source_input.is_symlink():
        raise ValueError(f"Sphinx build directory cannot be a symbolic link: {source_input}.")
    if destination_input.is_symlink():
        raise ValueError(f"Documentation site cannot be a symbolic link: {destination_input}.")
    source = source_input.resolve()
    destination = destination_input.resolve()

    if not source.is_dir() or not (source / "index.html").is_file():
        raise ValueError(f"Sphinx build directory has no index.html: {source}.")
    if _paths_overlap(source, destination):
        raise ValueError("The Sphinx build and GitHub Pages directories must not overlap.")
    _validate_root_tree(source)
    versions = _validate_existing_site(destination)

    destination.mkdir(parents=True, exist_ok=True)
    version_root = _version_root(destination)
    version_root.mkdir(exist_ok=True)
    target = version_root / series
    if target.parent != version_root:
        raise ValueError(f"Unsafe documentation series target: {target}.")

    _replace_series_tree(source, target)
    _, series_key = _canonical_series(series)
    versions = [(published, key) for published, key in versions if published != series]
    versions.append((series, series_key))
    versions.sort(key=lambda item: item[1], reverse=True)
    if versions[0][0] == series:
        _refresh_documentation_root(destination, normalized_url, versions)
    else:
        _write_site_metadata(destination, normalized_url, versions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage-series", help="replace one mutable major/minor documentation series")
    stage.add_argument("--build-dir", type=Path, required=True, help="completed Sphinx HTML build")
    stage.add_argument("--site-dir", type=Path, required=True, help="staged GitHub Pages site")
    stage.add_argument("--package-version", required=True, help="canonical stable x.y.z package version")
    stage.add_argument(
        "--base-url",
        default="https://dxogrp.github.io/blvpy/",
        help="public documentation URL without a version suffix",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        stage_documentation_series(args.build_dir, args.site_dir, args.package_version, args.base_url)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
