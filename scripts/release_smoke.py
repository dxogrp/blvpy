"""Smoke-test an installed BLVPY release distribution with native solvers."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import numpy as np

import blvpy as bp


def main() -> int:
    expected_version = os.environ["BLVPY_RELEASE_VERSION"]
    repository_root = Path(os.environ["BLVPY_REPOSITORY_ROOT"]).resolve()
    installed_package = Path(bp.__file__).resolve()
    if installed_package.is_relative_to(repository_root / "src"):
        raise RuntimeError(f"Smoke test imported the working tree instead of the distribution: {installed_package}.")
    if bp.__version__ != expected_version:
        raise RuntimeError(f"Installed BLVPY version is {bp.__version__!r}, expected {expected_version!r}.")

    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = bp.LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = bp.BilevelProblem(cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)), lower)
    result = problem.solve(epsilon_initial=1e-2, epsilon_target=1e-4, verbose=False)
    diagnostics = problem.gap_diagnostics(result)

    if not result.succeeded:
        raise RuntimeError(result.message or f"Release smoke solve ended with status {result.status!r}.")
    values = np.array([result.variable_values[x], result.variable_values[y]], dtype=float)
    if not np.isfinite(values).all() or result.residuals is None or result.residuals.max_violation > 1e-4:
        raise RuntimeError("Release smoke solve returned invalid values or residuals.")
    if diagnostics.source_gap is None or not np.isfinite(diagnostics.source_gap):
        raise RuntimeError("Release smoke diagnostics did not return a finite source gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
