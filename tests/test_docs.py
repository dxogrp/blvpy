from __future__ import annotations

import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import blvpy
import scripts.export_examples as export_module
from scripts.export_examples import export_examples
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


def _configuration() -> dict[str, object]:
    return runpy.run_path(str(DOCS_ROOT / "conf.py"))


def _make_build(directory: Path, marker: str, *, obsolete: bool = False) -> Path:
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(marker, encoding="utf-8")
    assets = directory / "_static"
    assets.mkdir()
    (assets / "current.css").write_text(f"/* {marker} */", encoding="utf-8")
    examples = directory / "examples"
    examples.mkdir()
    (examples / "demo.html").write_text(f"example: {marker}", encoding="utf-8")
    if obsolete:
        (directory / "obsolete.html").write_text("obsolete", encoding="utf-8")
        (assets / "obsolete.css").write_text("obsolete", encoding="utf-8")
        (examples / "retired.html").write_text("obsolete", encoding="utf-8")
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


def _write_notebook(directory: Path, name: str = "example.py") -> Path:
    directory.mkdir(exist_ok=True)
    notebook = directory / name
    notebook.write_text(f"print({name!r})\n", encoding="utf-8")
    return notebook


def _write_fake_export(command: list[str]) -> subprocess.CompletedProcess[str]:
    Path(command[-1]).write_text(Path(command[-3]).stem, encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_every_public_export_has_an_explicit_autodoc_entry() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    documented = set(AUTODOC_PATTERN.findall(documentation))

    assert documented == set(blvpy.__all__)


def test_every_example_is_linked_from_the_gallery() -> None:
    documentation = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_ROOT.rglob("*.md"))
    examples = sorted(path.stem for path in EXAMPLES_ROOT.glob("*.py"))
    linked_examples = sorted(EXAMPLE_ROLE_PATTERN.findall(documentation))

    assert examples
    assert linked_examples == examples


def test_documentation_uses_series_chrome_and_relative_example_links() -> None:
    configuration = _configuration()
    series = ".".join(blvpy.__version__.split(".")[:2])

    assert configuration["package_version"] == blvpy.__version__
    assert configuration["documentation_series"] == series
    assert configuration["version"] == series
    assert configuration["release"] == series
    assert configuration["html_title"] == f"BLVPY {series}"
    assert configuration["html_baseurl"] == f"https://dxogrp.github.io/blvpy/version/{series}/"
    assert configuration["extlinks"]["example"] == ("examples/%s.html", "%s")
    assert configuration["html_context"]["docs_switcher_url"] == "https://dxogrp.github.io/blvpy/switcher.json"
    assert "github_version" not in configuration["html_context"]


def test_only_example_links_open_in_a_new_tab() -> None:
    callback = _configuration()["_open_example_links_in_new_tab"]
    example = SimpleNamespace(attributes={"classes": ["extlink-example"]})
    ordinary = SimpleNamespace(attributes={"classes": []})
    doctree = SimpleNamespace(findall=lambda: (example, ordinary))

    callback(SimpleNamespace(builder=SimpleNamespace(format="html")), doctree, "examples")

    assert example.attributes["target"] == "_blank"
    assert example.attributes["rel"] == "noopener"
    assert "target" not in ordinary.attributes


def test_export_examples_isolated_and_replaces_stale_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_notebook(source, "b.py")
    _write_notebook(source, "a.py")
    (source / "zhlatex.mplstyle").write_bytes(b"axes.grid: True\n")
    generated = source / "figures"
    generated.mkdir()
    (generated / "existing.pdf").write_bytes(b"existing")
    source_before = _snapshot(source)
    output = tmp_path / "rendered"
    output.mkdir()
    (output / "retired.html").write_text("stale", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        working_directory = Path(kwargs["cwd"])
        if not (working_directory / "figures").exists():
            assert {path.name for path in working_directory.iterdir()} == {
                "a.py",
                "b.py",
                "zhlatex.mplstyle",
            }
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 180
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["MPLBACKEND"] == "Agg"
        assert environment["OMP_NUM_THREADS"] == environment["OPENBLAS_NUM_THREADS"] == "1"
        assert environment["PYTHONOPTIMIZE"] == "0"
        (working_directory / "figures").mkdir(exist_ok=True)
        (working_directory / "__marimo__").mkdir(exist_ok=True)
        return _write_fake_export(command)

    export_examples(source, output, runner=fake_runner)

    assert [Path(command[-3]).stem for command in calls] == ["a", "b"]
    assert calls[0][:5] == [sys.executable, "-m", "marimo", "export", "html"]
    assert {"--include-code", "--no-sandbox", "--force"} <= set(calls[0])
    assert sorted(path.name for path in output.iterdir()) == ["a.html", "b.html"]
    assert _snapshot(source) == source_before


def test_export_failure_preserves_previous_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_notebook(source)
    output = tmp_path / "rendered"
    output.mkdir()
    (output / "current.html").write_text("current", encoding="utf-8")
    before = _snapshot(output)

    def failing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        raise subprocess.CalledProcessError(2, command, stderr="export error")

    with pytest.raises(RuntimeError, match=r"example\.py"):
        export_examples(source, output, runner=failing_runner)

    assert _snapshot(output) == before
    assert not any(path.name.startswith(".rendered.") for path in tmp_path.iterdir())


def test_export_promotion_failure_restores_previous_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_notebook(source)
    output = tmp_path / "rendered"
    output.mkdir()
    (output / "current.html").write_text("current", encoding="utf-8")
    before = _snapshot(output)

    def successful_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return _write_fake_export(command)

    real_replace = export_module.os.replace
    replace_calls = 0

    def fail_promotion(source_path: Path, destination_path: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated promotion failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(export_module.os, "replace", fail_promotion)
    with pytest.raises(OSError, match="simulated promotion failure"):
        export_examples(source, output, runner=successful_runner)

    assert _snapshot(output) == before
    assert not any(path.name.startswith(".rendered.") for path in tmp_path.iterdir())


def test_export_rejects_unsafe_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "rendered"
    with pytest.raises(ValueError, match="no top-level Python notebooks"):
        export_examples(source, output)

    _write_notebook(source)
    with pytest.raises(ValueError, match="must not overlap"):
        export_examples(source, source / "rendered")

    output_target = tmp_path / "output-target"
    output_target.mkdir()
    output.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(ValueError, match="output directory cannot be a symbolic link"):
        export_examples(source, output)


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
    assert (site / "version" / "0.3" / "examples" / "demo.html").read_text(encoding="utf-8") == ("example: series 0.3")
    assert (site / "examples" / "demo.html").read_text(encoding="utf-8") == "example: series 0.3"
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
    assert not (series / "examples" / "retired.html").exists()
    assert not (site / "obsolete.html").exists()
    assert not (site / "_static" / "obsolete.css").exists()
    assert not (site / "examples" / "retired.html").exists()

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
