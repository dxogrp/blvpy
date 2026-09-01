from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import blvpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def _configuration(monkeypatch: pytest.MonkeyPatch, source_ref: str | None = None) -> dict[str, object]:
    if source_ref is None:
        monkeypatch.delenv("BLVPY_DOCS_SOURCE_REF", raising=False)
    else:
        monkeypatch.setenv("BLVPY_DOCS_SOURCE_REF", source_ref)
    return runpy.run_path(str(DOCS_ROOT / "conf.py"))


@pytest.mark.parametrize(
    ("package_version", "expected"),
    [
        ("0.3.0", "0.3"),
        ("0.3.1", "0.3"),
        ("0.10.12", "0.10"),
        ("12.4.0", "12.4"),
    ],
)
def test_documentation_series_uses_major_and_minor(
    monkeypatch: pytest.MonkeyPatch,
    package_version: str,
    expected: str,
) -> None:
    configuration = _configuration(monkeypatch)

    assert configuration["_documentation_series"](package_version) == expected


@pytest.mark.parametrize(
    "package_version",
    ["0.3", "0.3.0rc1", "0.3.0.post1", "0.3.0+local", "0.03.0", "v0.3.0", "1!0.3.0"],
)
def test_documentation_series_rejects_noncanonical_or_unstable_versions(
    monkeypatch: pytest.MonkeyPatch,
    package_version: str,
) -> None:
    configuration = _configuration(monkeypatch)

    with pytest.raises(RuntimeError, match="canonical stable major.minor.patch"):
        configuration["_documentation_series"](package_version)


def test_local_documentation_uses_series_chrome_and_exact_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = _configuration(monkeypatch)
    series = ".".join(blvpy.__version__.split(".")[:2])
    source_ref = f"v{blvpy.__version__}"
    example_url = f"https://github.com/dxogrp/blvpy/blob/{source_ref}/examples/%s"

    assert configuration["package_version"] == blvpy.__version__
    assert configuration["documentation_series"] == series
    assert configuration["version"] == series
    assert configuration["release"] == series
    assert configuration["html_title"] == f"BLVPY {series}"
    assert configuration["html_baseurl"] == f"https://dxogrp.github.io/blvpy/version/{series}/"
    assert configuration["extlinks"]["example"][0] == example_url
    assert configuration["html_context"]["github_version"] == source_ref

    template = (DOCS_ROOT / "_templates" / "versions.html").read_text(encoding="utf-8")
    assert 'data-current-version="{{ version }}"' in template
    assert "Release {{ version }}" in template
    assert "{{ release }}" not in template


def test_deployed_documentation_uses_exact_source_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    source_ref = "0123456789abcdef0123456789abcdef01234567"
    configuration = _configuration(monkeypatch, source_ref)
    example_url = f"https://github.com/dxogrp/blvpy/blob/{source_ref}/examples/%s"

    assert configuration["docs_source_ref"] == source_ref
    assert configuration["html_context"]["github_version"] == source_ref
    assert configuration["extlinks"]["example"][0] == example_url
