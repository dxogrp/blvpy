"""Publish latest and immutable BLVPY documentation on GitHub Pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from packaging.version import InvalidVersion, Version

VERSION_DIRECTORY = "version"


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


def _tree_manifest(root: Path) -> tuple[tuple[str, str, str | None], ...]:
    """Return a content manifest while rejecting unsafe filesystem entries."""
    entries: list[tuple[str, str, str | None]] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Documentation trees cannot contain symbolic links: {path}.")
        if path.is_dir():
            entries.append(("directory", relative, None))
        elif path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entries.append(("file", relative, digest))
        else:
            raise ValueError(f"Unsupported documentation filesystem entry: {path}.")
    return tuple(entries)


def _validate_root_tree(root: Path) -> None:
    """Validate a documentation tree before copying it to the site root."""
    _tree_manifest(root)
    for path in root.iterdir():
        if path.name in {".git", VERSION_DIRECTORY}:
            raise ValueError(f"Documentation tree contains a reserved site entry: {path}.")
        try:
            _canonical_version(path.name)
        except ValueError:
            continue
        raise ValueError(f"Documentation root entry conflicts with a version directory: {path}.")


def _install_immutable_tree(source: Path, target: Path) -> None:
    """Install a new version tree, allowing only identical release reruns."""
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(f"Documentation version target is not a safe directory: {target}.")
    source_manifest = _tree_manifest(source)
    if target.exists():
        if _tree_manifest(target) == source_manifest:
            return
        raise ValueError(f"Documentation version {target.name!r} is already published with different content.")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _version_root(site_dir: Path) -> Path:
    root = site_dir / VERSION_DIRECTORY
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"Documentation version root is not a safe directory: {root}.")
    return root


def _published_versions(site_dir: Path) -> list[tuple[str, Version]]:
    root = _version_root(site_dir)
    if not root.exists():
        return []

    versions: list[tuple[str, Version]] = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            canonical, parsed = _canonical_version(path.name)
        except ValueError:
            continue
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


def _write_site_metadata(site_dir: Path, base_url: str) -> None:
    versions = _published_versions(site_dir)
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
    """Remove mutable root documentation while preserving immutable releases."""
    git_metadata = site_dir / ".git"
    if git_metadata.is_symlink():
        raise ValueError(f"Documentation sites cannot contain symbolic links: {git_metadata}.")
    preserved_names = {VERSION_DIRECTORY, ".git"}
    root_entries = [path for path in site_dir.iterdir() if path.name not in preserved_names]
    for path in root_entries:
        if path.is_symlink():
            raise ValueError(f"Documentation sites cannot contain symbolic links: {path}.")
        if path.is_dir():
            _tree_manifest(path)
        elif not path.is_file():
            raise ValueError(f"Unsupported documentation filesystem entry: {path}.")

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


def _refresh_documentation_root(site_dir: Path, base_url: str) -> None:
    """Serve the greatest published version directly from the Pages root."""
    versions = _published_versions(site_dir)
    if not versions:
        raise ValueError("The staged site does not contain a released documentation version.")

    version_root = _version_root(site_dir)
    for version, _ in versions:
        _tree_manifest(version_root / version)

    preferred = version_root / versions[0][0]
    _validate_root_tree(preferred)
    _remove_root_documentation(site_dir)
    _copy_tree_contents(preferred, site_dir)
    _write_site_metadata(site_dir, base_url)


def stage_documentation(build_dir: Path, site_dir: Path, version: str, base_url: str) -> None:
    """Stage a release and serve the greatest published version at the root."""
    canonical, _ = _canonical_version(version)
    normalized_url = _canonical_base_url(base_url)
    source = build_dir.expanduser().resolve()
    destination = site_dir.expanduser().resolve()

    if not source.is_dir() or not (source / "index.html").is_file():
        raise ValueError(f"Sphinx build directory has no index.html: {source}.")
    if _paths_overlap(source, destination):
        raise ValueError("The Sphinx build and GitHub Pages directories must not overlap.")
    _validate_root_tree(source)

    destination.mkdir(parents=True, exist_ok=True)
    version_root = _version_root(destination)
    version_root.mkdir(exist_ok=True)
    target = version_root / canonical
    if target.parent != version_root:
        raise ValueError(f"Unsafe documentation version target: {target}.")

    _install_immutable_tree(source, target)
    _refresh_documentation_root(destination, normalized_url)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage-release", help="install one immutable release and refresh the latest root")
    stage.add_argument("--build-dir", type=Path, required=True, help="completed Sphinx HTML build")
    stage.add_argument("--site-dir", type=Path, required=True, help="checked-out gh-pages worktree")
    stage.add_argument("--version", required=True, help="canonical PEP 440 release version")
    stage.add_argument(
        "--base-url",
        default="https://dxogrp.github.io/blvpy/",
        help="public documentation URL without a version suffix",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        stage_documentation(args.build_dir, args.site_dir, args.version, args.base_url)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
