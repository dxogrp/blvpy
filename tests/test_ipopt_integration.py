"""Native IPOPT end-to-end checks.

These tests are marked because they execute against the native IPOPT library.
The Linux CI jobs install IPOPT before running the complete suite.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy import BilevelProblem, LowerProblem

pytestmark = pytest.mark.ipopt


def _require_ipopt() -> None:
    if "IPOPT" not in {str(solver).upper() for solver in cp.installed_solvers()}:
        pytest.skip("IPOPT/cyipopt is not installed")


def test_analytic_quadratic_reaches_target_and_matches_direct_lower_solve() -> None:
    _require_ipopt()
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
    )

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        starts=3,
        seed=4,
        solver_options={"hessian_approximation": "limited-memory", "tol": 1e-8},
    )

    assert result.succeeded
    assert result.final_epsilon == pytest.approx(1e-5)
    assert all(left > right for left, right in zip(result.epsilon_history, result.epsilon_history[1:]))
    assert result.residuals is not None
    assert result.residuals.max_violation <= 1e-5
    assert float(x.value) == pytest.approx(0.0, abs=2e-3)
    assert float(y.value) == pytest.approx(0.0, abs=2e-3)

    returned_y = float(y.value)
    parameter = next(iter(model._parameter_links))
    parameter.value = float(x.value)
    model._cvxpy_lower_problem.solve(solver=cp.CLARABEL)
    assert model._cvxpy_lower_problem.status in cp.settings.SOLUTION_PRESENT
    assert returned_y == pytest.approx(float(y.value), abs=2e-4)


def test_optimistic_lp_selects_upper_preferred_lower_optimizer() -> None:
    _require_ipopt()
    x = cp.Variable(name="x", bounds=[0.0, 1.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(0.0 * y), [y >= x, y <= 1.0], parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
        lower,
    )

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        starts=3,
        seed=11,
        solver_options={"hessian_approximation": "limited-memory", "tol": 1e-8},
    )

    assert result.succeeded
    np.testing.assert_allclose([x.value, y.value], [0.0, 1.0], atol=3e-3)
