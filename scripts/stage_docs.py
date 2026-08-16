"""Stage one immutable BLVPY documentation release for GitHub Pages."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from packaging.version import InvalidVersion, Version


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


def _published_versions(site_dir: Path) -> list[tuple[str, Version]]:
    versions: list[tuple[str, Version]] = []
    for path in site_dir.iterdir():
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

    entries = [
        {
            "name": version,
            "version": version,
            "url": f"{base_url}/{version}/",
            "preferred": index == 0,
        }
        for index, (version, _) in enumerate(versions)
    ]
    _write_text_atomic(site_dir / "switcher.json", json.dumps(entries, indent=2) + "\n")
    _write_text_atomic(site_dir / ".nojekyll", "")

    preferred_url = entries[0]["url"]
    escaped_url = html.escape(preferred_url, quote=True)
    redirect = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={escaped_url}">
    <link rel="canonical" href="{escaped_url}">
    <title>BLVPY documentation</title>
  </head>
  <body>
    <p>Redirecting to <a href="{escaped_url}">BLVPY {html.escape(entries[0]["version"])} documentation</a>.</p>
  </body>
</html>
"""
    _write_text_atomic(site_dir / "index.html", redirect)


def stage_documentation(build_dir: Path, site_dir: Path, version: str, base_url: str) -> None:
    """Stage a Sphinx HTML build as one version of the GitHub Pages site."""
    canonical, _ = _canonical_version(version)
    normalized_url = _canonical_base_url(base_url)
    source = build_dir.expanduser().resolve()
    destination = site_dir.expanduser().resolve()

    if not source.is_dir() or not (source / "index.html").is_file():
        raise ValueError(f"Sphinx build directory has no index.html: {source}.")
    if _paths_overlap(source, destination):
        raise ValueError("The Sphinx build and GitHub Pages directories must not overlap.")

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / canonical
    if target.parent != destination:
        raise ValueError(f"Unsafe documentation version target: {target}.")

    _install_immutable_tree(source, target)
    _write_site_metadata(destination, normalized_url)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True, help="completed Sphinx HTML build")
    parser.add_argument("--site-dir", type=Path, required=True, help="checked-out gh-pages worktree")
    parser.add_argument("--version", required=True, help="canonical PEP 440 release version")
    parser.add_argument(
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
