"""Validate BLVPY release distributions and write their SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tomllib
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

_PROJECT_NAME = "blvpy"
_PROHIBITED_SDIST_PATHS = {
    ".github",
    "Makefile",
    "docs",
    "examples",
    "scripts",
    "tests",
    "uv.lock",
}
_GENERATED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def canonical_stable_version(value: str) -> Version:
    """Parse a canonical stable public version or raise ``ValueError``."""
    try:
        version = Version(value)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid release version: {value!r}.") from exc
    if str(version) != value:
        raise ValueError(f"Release version must use canonical PEP 440 spelling: {value!r}.")
    if (
        version.epoch != 0
        or len(version.release) != 3
        or version.is_prerelease
        or version.is_devrelease
        or version.local is not None
    ):
        raise ValueError(f"Release version must be stable and public: {value!r}.")
    return version


def _distribution_files(dist_dir: Path) -> tuple[Path, Path]:
    distributions = sorted(
        path
        for path in dist_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    wheels = [path for path in distributions if path.suffix == ".whl"]
    sdists = [path for path in distributions if path.name.endswith(".tar.gz")]
    if len(distributions) != 2 or len(wheels) != 1 or len(sdists) != 1:
        names = ", ".join(path.name for path in distributions) or "none"
        raise ValueError(f"Expected exactly one wheel and one sdist; found: {names}.")
    return wheels[0], sdists[0]


def _validate_archive_path(name: str, *, archive: Path) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in {archive.name}: {name!r}.")
    if any(part in _GENERATED_PARTS or part.endswith((".pyc", ".pyo")) for part in path.parts):
        raise ValueError(f"Generated file in {archive.name}: {name!r}.")
    return path


def _metadata_value(raw: bytes, field: str, *, archive: Path) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    value = message.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{archive.name} metadata is missing {field!r}.")
    return value


def _validate_wheel(wheel: Path, expected_version: Version, source_root: Path) -> None:
    name, version, build, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != _PROJECT_NAME or version != expected_version or build:
        raise ValueError(f"Unexpected wheel filename: {wheel.name!r}.")
    if {str(tag) for tag in tags} != {"py3-none-any"}:
        raise ValueError(f"BLVPY must publish one pure-Python py3-none-any wheel, not {wheel.name!r}.")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        paths = [_validate_archive_path(name, archive=wheel) for name in names]
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError(f"{wheel.name} must contain exactly one METADATA and one WHEEL file.")

        metadata = archive.read(metadata_names[0])
        if canonicalize_name(_metadata_value(metadata, "Name", archive=wheel)) != _PROJECT_NAME:
            raise ValueError(f"{wheel.name} contains the wrong project name.")
        if Version(_metadata_value(metadata, "Version", archive=wheel)) != expected_version:
            raise ValueError(f"{wheel.name} contains the wrong project version.")
        if _metadata_value(metadata, "Requires-Python", archive=wheel) != ">=3.12":
            raise ValueError(f"{wheel.name} contains an unexpected Python requirement.")

        wheel_metadata = archive.read(wheel_names[0])
        if _metadata_value(wheel_metadata, "Root-Is-Purelib", archive=wheel).lower() != "true":
            raise ValueError(f"{wheel.name} is not marked as a pure-Python wheel.")
        if _metadata_value(wheel_metadata, "Tag", archive=wheel) != "py3-none-any":
            raise ValueError(f"{wheel.name} contains an unexpected compatibility tag.")

        archived = {path.as_posix() for path in paths}
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in archived):
            raise ValueError(f"{wheel.name} is missing its license file.")
        for source in source_root.glob("*.py"):
            expected = f"blvpy/{source.name}"
            if expected not in archived:
                raise ValueError(f"{wheel.name} is missing package source {expected!r}.")


def _validate_sdist(sdist: Path, expected_version: Version, source_root: Path) -> None:
    name, version = parse_sdist_filename(sdist.name)
    if canonicalize_name(name) != _PROJECT_NAME or version != expected_version:
        raise ValueError(f"Unexpected sdist filename: {sdist.name!r}.")

    expected_root = f"{_PROJECT_NAME}-{expected_version}"
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError(f"{sdist.name} is empty.")
        paths = [_validate_archive_path(member.name, archive=sdist) for member in members]
        if any(member.issym() or member.islnk() or not (member.isfile() or member.isdir()) for member in members):
            raise ValueError(f"{sdist.name} contains unsupported filesystem entries.")
        if any(not path.parts or path.parts[0] != expected_root for path in paths):
            raise ValueError(f"{sdist.name} must contain only the root directory {expected_root!r}.")

        relative = {PurePosixPath(*path.parts[1:]).as_posix() for path in paths if len(path.parts) > 1}
        required = {"LICENSE", "PKG-INFO", "README.md", "pyproject.toml"}
        required.update(f"src/blvpy/{source.name}" for source in source_root.glob("*.py"))
        missing = sorted(required - relative)
        if missing:
            raise ValueError(f"{sdist.name} is missing required files: {', '.join(missing)}.")

        for prohibited in _PROHIBITED_SDIST_PATHS:
            if any(path == prohibited or path.startswith(f"{prohibited}/") for path in relative):
                raise ValueError(f"{sdist.name} contains development-only path {prohibited!r}.")

        pyproject_member = archive.getmember(f"{expected_root}/pyproject.toml")
        extracted = archive.extractfile(pyproject_member)
        if extracted is None:
            raise ValueError(f"Could not read pyproject.toml from {sdist.name}.")
        project = tomllib.loads(extracted.read().decode("utf-8"))["project"]
        if canonicalize_name(project["name"]) != _PROJECT_NAME or Version(project["version"]) != expected_version:
            raise ValueError(f"{sdist.name} contains inconsistent project metadata.")

        package_info_member = archive.getmember(f"{expected_root}/PKG-INFO")
        package_info_file = archive.extractfile(package_info_member)
        if package_info_file is None:
            raise ValueError(f"Could not read PKG-INFO from {sdist.name}.")
        package_info = package_info_file.read()
        if canonicalize_name(_metadata_value(package_info, "Name", archive=sdist)) != _PROJECT_NAME:
            raise ValueError(f"{sdist.name} contains the wrong project name.")
        if Version(_metadata_value(package_info, "Version", archive=sdist)) != expected_version:
            raise ValueError(f"{sdist.name} contains the wrong project version.")
        if _metadata_value(package_info, "Requires-Python", archive=sdist) != ">=3.12":
            raise ValueError(f"{sdist.name} contains an unexpected Python requirement.")


def verify_distributions(dist_dir: Path, version: str, source_root: Path) -> tuple[Path, Path]:
    """Validate and return the wheel and sdist for one BLVPY release."""
    expected_version = canonical_stable_version(version)
    directory = dist_dir.resolve()
    sources = source_root.resolve()
    if not directory.is_dir():
        raise ValueError(f"Distribution directory does not exist: {directory}.")
    if not sources.is_dir():
        raise ValueError(f"Package source directory does not exist: {sources}.")

    wheel, sdist = _distribution_files(directory)
    _validate_wheel(wheel, expected_version, sources)
    _validate_sdist(sdist, expected_version, sources)
    return wheel, sdist


def write_checksums(distributions: tuple[Path, Path], output: Path) -> None:
    """Write stable SHA-256 checksum lines for release distributions."""
    lines = []
    for path in sorted(distributions, key=lambda item: item.name):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {path.name}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", type=Path, default=Path("dist"), help="directory containing release distributions"
    )
    parser.add_argument("--version", required=True, help="canonical stable package version")
    parser.add_argument("--source-root", type=Path, default=Path("src/blvpy"), help="BLVPY package source directory")
    parser.add_argument("--checksums", type=Path, default=Path("dist/SHA256SUMS"), help="checksum output file")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        distributions = verify_distributions(args.dist_dir, args.version, args.source_root)
        write_checksums(distributions, args.checksums)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
