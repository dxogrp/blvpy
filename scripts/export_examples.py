"""Execute Marimo examples and transactionally export their static HTML pages."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _source_files(source_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return validated top-level notebooks and style files."""
    notebooks: list[Path] = []
    styles: list[Path] = []
    for path in sorted(source_dir.iterdir(), key=lambda entry: entry.name):
        if path.is_symlink():
            raise ValueError(f"Example sources cannot contain symbolic links: {path}.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Unsupported example filesystem entry: {path}.")
        if path.suffix == ".py":
            notebooks.append(path)
        elif path.suffix == ".mplstyle":
            styles.append(path)

    if not notebooks:
        raise ValueError(f"Example source directory contains no top-level Python notebooks: {source_dir}.")
    return notebooks, styles


def _format_log(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _failure_message(notebook: Path, returncode: int, stdout: Any, stderr: Any) -> str:
    message = f"Marimo export failed for {notebook.name} with exit status {returncode}."
    captured = []
    if formatted := _format_log(stdout):
        captured.append(f"stdout:\n{formatted}")
    if formatted := _format_log(stderr):
        captured.append(f"stderr:\n{formatted}")
    if captured:
        message = f"{message}\n" + "\n".join(captured)
    return message


def _run_export(
    notebook: Path,
    destination: Path,
    *,
    runner: Runner,
    timeout: float,
    working_directory: Path,
    environment: dict[str, str],
) -> None:
    command = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        "--include-code",
        "--no-sandbox",
        "--force",
        str(notebook),
        "--output",
        str(destination),
    ]
    try:
        completed = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_directory,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Marimo export timed out after {timeout:g} seconds for {notebook.name}.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_failure_message(notebook, exc.returncode, exc.stdout, exc.stderr)) from exc

    if completed.returncode != 0:
        raise RuntimeError(_failure_message(notebook, completed.returncode, completed.stdout, completed.stderr))
    if destination.is_symlink() or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Marimo export did not produce a nonempty HTML file for {notebook.name}: {destination}.")


def _replace_directory(staging: Path, destination: Path) -> None:
    """Atomically promote a complete staging directory, preserving the old tree on failure."""
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"Example output must be a directory and cannot be a symbolic link: {destination}.")

    backup: Path | None = None
    if destination.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
        backup.rmdir()

    try:
        if backup is not None:
            os.replace(destination, backup)
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        raise
    else:
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                print(
                    f"Rendered examples were updated, but the previous output could not be removed: {exc}",
                    file=sys.stderr,
                )


def export_examples(
    source_dir: Path,
    output_dir: Path,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 180,
) -> None:
    """Execute every top-level Marimo notebook and replace ``output_dir`` on success."""
    if timeout <= 0:
        raise ValueError("The Marimo export timeout must be positive.")

    source_input = source_dir.expanduser()
    output_input = output_dir.expanduser()
    if source_input.is_symlink():
        raise ValueError(f"Example source directory cannot be a symbolic link: {source_input}.")
    if output_input.is_symlink():
        raise ValueError(f"Example output directory cannot be a symbolic link: {output_input}.")

    source = source_input.resolve()
    output = output_input.resolve()
    if not source.is_dir():
        raise ValueError(f"Example source directory does not exist: {source}.")
    if _paths_overlap(source, output):
        raise ValueError("Example source and output directories must not overlap.")
    notebooks, styles = _source_files(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError(f"Example output parent must be a directory and cannot be a symbolic link: {output.parent}.")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"Example output must be a directory and cannot be a symbolic link: {output}.")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        with (
            tempfile.TemporaryDirectory(prefix="blvpy-example-input-") as temporary_name,
            tempfile.TemporaryDirectory(prefix="blvpy-matplotlib-") as matplotlib_config_name,
        ):
            temporary_input = Path(temporary_name)
            for source_file in [*notebooks, *styles]:
                shutil.copy2(source_file, temporary_input / source_file.name)

            environment = os.environ.copy()
            environment.update(
                {
                    "BLIS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": matplotlib_config_name,
                    "NUMEXPR_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                }
            )

            for source_notebook in notebooks:
                notebook = temporary_input / source_notebook.name
                destination = staging / f"{source_notebook.stem}.html"
                _run_export(
                    notebook,
                    destination,
                    runner=runner,
                    timeout=timeout,
                    working_directory=temporary_input,
                    environment=environment,
                )

        _replace_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="directory containing Marimo example sources")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory to replace with exported HTML pages")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        export_examples(args.source_dir, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
