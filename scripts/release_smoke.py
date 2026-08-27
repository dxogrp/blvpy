"""Smoke-test an installed BLVPY release distribution with native solvers."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import numpy as np

import blvpy as bp


def _smoke_minimize() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = bp.LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = bp.BilevelProblem(cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)), lower)
    result = problem.solve(epsilon_initial=1e-2, epsilon_target=1e-4, verbose=False)

    if not result.succeeded:
        raise RuntimeError(result.message or f"Release smoke solve ended with status {result.status!r}.")
    diagnostics = problem.gap_diagnostics(result)
    values = np.array([result.variable_values[x], result.variable_values[y]], dtype=float)
    if not np.isfinite(values).all() or result.residuals is None or result.residuals.max_violation > 1e-4:
        raise RuntimeError("Release smoke solve returned invalid values or residuals.")
    if diagnostics.source_gap is None or not np.isfinite(diagnostics.source_gap):
        raise RuntimeError("Release smoke diagnostics did not return a finite source gap.")


def _smoke_maximize() -> None:
    x = cp.Variable(name="max_x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="max_y")
    lower_objective = cp.Maximize(3.0 - cp.square(y - x))
    lower = bp.LowerProblem(lower_objective, parameters=[x])
    problem = bp.BilevelProblem(
        cp.Maximize(7.0 - cp.square(x - 0.5) - cp.square(y - 0.5)),
        lower,
    )
    result = problem.solve(epsilon_initial=1e-2, epsilon_target=1e-4, seed=5, verbose=False)

    if not result.succeeded:
        raise RuntimeError(result.message or f"Maximize smoke solve ended with status {result.status!r}.")
    diagnostics = problem.gap_diagnostics(result)
    values = np.array([result.variable_values[x], result.variable_values[y]], dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, [0.5, 0.5], atol=2e-3, rtol=0.0):
        raise RuntimeError(f"Maximize smoke solve returned unexpected variable values: {values}.")
    if result.final_epsilon is None or not np.isclose(result.final_epsilon, 1e-4, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Maximize smoke solve stopped at epsilon {result.final_epsilon!r}, expected 1e-4.")
    if result.objective is None or not np.isclose(result.objective, 7.0, atol=2e-3, rtol=0.0):
        raise RuntimeError(f"Maximize smoke solve returned upper objective {result.objective!r}, expected 7.0.")
    if result.selected_run is None or not np.isclose(result.selected_run.objective, 7.0, atol=2e-3, rtol=0.0):
        raise RuntimeError("Maximize smoke selected run did not retain the original-sense objective.")
    if result.final_iteration is None or not np.isclose(result.final_iteration.objective, 7.0, atol=2e-3, rtol=0.0):
        raise RuntimeError("Maximize smoke final iteration did not retain the original-sense objective.")
    if lower_objective.value is None or not np.isclose(lower_objective.value, 3.0, atol=2e-3, rtol=0.0):
        raise RuntimeError(f"Maximize smoke solve returned lower objective {lower_objective.value!r}, expected 3.0.")
    if result.residuals is None or result.residuals.max_violation > 1e-4:
        raise RuntimeError("Maximize smoke solve returned invalid residuals.")
    if (
        diagnostics.source_gap is None
        or not np.isfinite(diagnostics.source_gap)
        or diagnostics.source_gap < -1e-7
        or diagnostics.source_gap > result.final_epsilon + 1e-6
    ):
        raise RuntimeError(f"Maximize smoke diagnostics returned invalid source gap {diagnostics.source_gap!r}.")


def main() -> int:
    expected_version = os.environ["BLVPY_RELEASE_VERSION"]
    repository_root = Path(os.environ["BLVPY_REPOSITORY_ROOT"]).resolve()
    installed_package = Path(bp.__file__).resolve()
    if installed_package.is_relative_to(repository_root / "src"):
        raise RuntimeError(f"Smoke test imported the working tree instead of the distribution: {installed_package}.")
    if bp.__version__ != expected_version:
        raise RuntimeError(f"Installed BLVPY version is {bp.__version__!r}, expected {expected_version!r}.")

    _smoke_minimize()
    _smoke_maximize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
