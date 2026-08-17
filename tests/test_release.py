from __future__ import annotations

import hashlib
import importlib.metadata
import io
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.verify_release import canonical_stable_version, verify_distributions, write_checksums

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTION_PATTERN = re.compile(r"^\s*uses:\s+([^\s#]+)", re.MULTILINE)


def _source_tree(root: Path) -> Path:
    source = root / "src" / "blvpy"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        'from importlib.metadata import version\n\n__version__ = version("blvpy")\n',
        encoding="utf-8",
    )
    (source / "problem.py").write_text("class BilevelProblem: ...\n", encoding="utf-8")
    return source


def _wheel(dist: Path, *, version: str = "0.1.0", metadata_version: str | None = None) -> Path:
    path = dist / f"blvpy-{version}-py3-none-any.whl"
    metadata = metadata_version or version
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "blvpy/__init__.py",
            'from importlib.metadata import version\n\n__version__ = version("blvpy")\n',
        )
        archive.writestr("blvpy/problem.py", "class BilevelProblem: ...\n")
        archive.writestr(
            f"blvpy-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: blvpy\nVersion: {metadata}\nRequires-Python: >=3.12\n\n",
        )
        archive.writestr(
            f"blvpy-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        )
        archive.writestr(f"blvpy-{version}.dist-info/licenses/LICENSE", "Apache License\n")
    return path


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _sdist(dist: Path, source: Path, *, version: str = "0.1.0", extra: str | None = None) -> Path:
    path = dist / f"blvpy-{version}.tar.gz"
    root = f"blvpy-{version}"
    pyproject = f'[project]\nname = "blvpy"\nversion = "{version}"\n'
    with tarfile.open(path, "w:gz") as archive:
        _add_tar_bytes(archive, f"{root}/LICENSE", b"Apache License\n")
        _add_tar_bytes(
            archive,
            f"{root}/PKG-INFO",
            f"Metadata-Version: 2.4\nName: blvpy\nVersion: {version}\nRequires-Python: >=3.12\n\n".encode(),
        )
        _add_tar_bytes(archive, f"{root}/README.md", b"# BLVPY\n")
        _add_tar_bytes(archive, f"{root}/pyproject.toml", pyproject.encode())
        for path_source in source.glob("*.py"):
            _add_tar_bytes(archive, f"{root}/src/blvpy/{path_source.name}", path_source.read_bytes())
        if extra is not None:
            _add_tar_bytes(archive, f"{root}/{extra}", b"unexpected\n")
    return path


@pytest.mark.parametrize("version", ["0.1.0", "1.2.3.post1"])
def test_canonical_stable_release_versions(version: str) -> None:
    assert str(canonical_stable_version(version)) == version


@pytest.mark.parametrize("version", ["v0.1.0", "0.1", "1!0.2.0", "0.2.0rc1", "0.2.0.dev1", "0.2.0+local"])
def test_noncanonical_or_unstable_release_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="release version|Release version"):
        canonical_stable_version(version)


def test_release_distributions_and_checksums(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = _wheel(dist)
    sdist = _sdist(dist, source)

    assert verify_distributions(dist, "0.1.0", source) == (wheel, sdist)

    checksums = dist / "SHA256SUMS"
    write_checksums((wheel, sdist), checksums)
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, sdist)}
    actual = {
        name: digest
        for digest, name in (line.split("  ", 1) for line in checksums.read_text(encoding="utf-8").splitlines())
    }
    assert actual == expected


def test_release_verifier_rejects_inconsistent_wheel_metadata(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, metadata_version="0.2.0")
    _sdist(dist, source)

    with pytest.raises(ValueError, match="wrong project version"):
        verify_distributions(dist, "0.1.0", source)


def test_release_verifier_rejects_development_files_in_sdist(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist)
    _sdist(dist, source, extra="tests/test_release.py")

    with pytest.raises(ValueError, match="development-only path 'tests'"):
        verify_distributions(dist, "0.1.0", source)


def test_hatch_sdist_manifest_is_minimal() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["hatch"]["build"]["targets"]["sdist"] == {
        "only-include": ["src/blvpy", "README.md", "LICENSE", "pyproject.toml"]
    }


def test_runtime_version_comes_from_distribution_metadata() -> None:
    import blvpy

    assert blvpy.__version__ == importlib.metadata.version("blvpy")


def test_release_workflows_are_ordered_and_action_pins_are_immutable() -> None:
    release = (REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    documentation = (REPOSITORY_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert "release:" in release and "- published" in release
    assert "needs: verify" in release
    assert "- publish" in release
    assert "- attach-assets" in release
    assert "uses: ./.github/workflows/docs.yml" in release
    assert "Publish stable and versioned documentation" in release
    assert "workflow_dispatch:" in documentation
    assert "workflow_call:" in documentation
    assert "scripts/stage_docs.py stage-release" in documentation
    assert "Rebuild single-version documentation site" in documentation
    assert "Refusing to rebuild documentation history" in documentation

    external_actions = [
        action for action in ACTION_PATTERN.findall(release + "\n" + documentation) if not action.startswith("./")
    ]
    assert external_actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in external_actions)
