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
WORKFLOWS_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_PATTERN = re.compile(r"^\s*(?:-\s*)?uses:\s+([^\s#]+)", re.MULTILINE)
# JavaScript actions use Node.js 24; composite actions transitively use Node.js 24 or Docker.
REVIEWED_ACTION_PINS = frozenset(
    {
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d",
        "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9",
        "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b",
        "conda-incubator/setup-miniconda@8ee1f361103df19b6f8c8655fd3967a8ecb162d5",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
    }
)


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


def test_release_and_manual_documentation_workflows_are_separated() -> None:
    release = (WORKFLOWS_DIRECTORY / "release.yml").read_text(encoding="utf-8")
    documentation = (WORKFLOWS_DIRECTORY / "docs.yml").read_text(encoding="utf-8")

    assert "release:" in release and "- published" in release
    assert "needs: verify" in release
    assert "- publish" in release
    assert "attach-assets:" in release
    assert "Check Sphinx documentation" in release
    assert "uses: ./.github/workflows/docs.yml" not in release
    assert "workflow_dispatch:" in documentation
    assert "workflow_call:" not in documentation
    assert "group: blvpy-pages" in documentation
    assert "cancel-in-progress: false" in documentation
    assert "ref: ${{ github.sha }}" in documentation
    assert "BLVPY_DOCS_SOURCE_REF: ${{ github.sha }}" in documentation
    assert "${{ github.run_attempt }}" in documentation
    assert "artifact_name: blvpy-pages-series-" in documentation
    assert "scripts/stage_docs.py stage-series" in documentation
    assert '--package-version "$PACKAGE_VERSION"' in documentation
    assert 'rsync --archive --checksum --delete --exclude=.git "$SITE_DIR/" "$PAGES_DIR/"' in documentation
    assert 'diff --recursive --brief --exclude=.git "$SITE_DIR" "$PAGES_DIR"' in documentation
    assert "stage-release" not in documentation
    assert "checkout --orphan" not in documentation
    assert "git push --force" not in documentation
    assert "git push -f" not in documentation
    pages_upload = documentation.split("      - name: Upload Pages artifact\n", 1)[1].split("\n\n", 1)[0]
    assert "include-hidden-files: true" in pages_upload


# ONE-TIME MIGRATION CONTRACT: delete this test with rebuild-docs-once.yml after
# the rebuilt live site and all three documentation series have been verified.
def test_one_time_documentation_rebuild_workflow_contract() -> None:
    rebuild = (WORKFLOWS_DIRECTORY / "rebuild-docs-once.yml").read_text(encoding="utf-8")

    assert "ONE-TIME MIGRATION WORKFLOW: DELETE THIS FILE" in rebuild
    assert "workflow_dispatch:" in rebuild
    assert "required: true" in rebuild
    assert '"REBUILD gh-pages"' in rebuild
    assert '"refs/heads/main"' in rebuild
    assert "group: blvpy-pages" in rebuild
    assert "cancel-in-progress: false" in rebuild

    assert "checkout_ref: v0.1.0" in rebuild
    assert "source_ref: v0.1.0" in rebuild
    assert "checkout_ref: v0.2.0" in rebuild
    assert "source_ref: v0.2.0" in rebuild
    assert "checkout_ref: ${{ github.sha }}" in rebuild
    assert "source_ref: ${{ github.sha }}" in rebuild
    assert "${{ github.run_attempt }}" in rebuild
    assert "artifact_name: blvpy-pages-rebuild-" in rebuild
    assert "cp policy/docs/conf.py source/docs/conf.py" in rebuild
    assert "policy/docs/_templates/" in rebuild
    assert "policy/docs/_static/version-switcher.js" in rebuild
    assert rebuild.count("uv sync --frozen --group docs") == 1
    assert rebuild.count("scripts/stage_docs.py stage-series") == 3

    assert 'expected_series = {"0.1", "0.2", "0.3"}' in rebuild
    assert '"name": "latest"' in rebuild
    assert "The documentation root does not exactly match series 0.3." in rebuild
    diagnostic_upload = rebuild.index("      - name: Upload fresh-site diagnostic artifact")
    pages_upload = rebuild.index("      - name: Upload Pages artifact before changing gh-pages")
    replacement = rebuild.index("      - name: Replace the existing gh-pages tree in a normal worktree")
    assert diagnostic_upload < replacement and pages_upload < replacement
    assert 'rsync --archive --checksum --delete --exclude=.git "$SITE_DIR/" "$PAGES_DIR/"' in rebuild
    assert 'diff --recursive --brief --exclude=.git "$SITE_DIR" "$PAGES_DIR"' in rebuild
    assert "Previous gh-pages tip" in rebuild
    assert "Replacement commit" in rebuild
    assert "Main source SHA" in rebuild
    assert "Site manifest SHA-256" in rebuild
    assert "checkout --orphan" not in rebuild
    assert "git push --force" not in rebuild
    assert "git push -f" not in rebuild


def test_external_workflow_action_pins_are_immutable() -> None:
    workflows = sorted([*WORKFLOWS_DIRECTORY.glob("*.yml"), *WORKFLOWS_DIRECTORY.glob("*.yaml")])
    assert workflows
    external_actions = [
        action
        for workflow in workflows
        for action in ACTION_PATTERN.findall(workflow.read_text(encoding="utf-8"))
        if not action.startswith("./")
    ]
    assert external_actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in external_actions)
    assert set(external_actions) == REVIEWED_ACTION_PINS
