"""Native IPOPT end-to-end checks.

These tests are kept separate because they execute against the native IPOPT
library. The Linux integration job installs IPOPT before selecting this mark.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy import BilevelProblem


pytestmark = pytest.mark.ipopt


def _require_ipopt() -> None:
    if "IPOPT" not in {str(solver).upper() for solver in cp.installed_solvers()}:
        pytest.skip("IPOPT/cyipopt is not installed")


def test_analytic_quadratic_reaches_target_and_matches_direct_lower_solve() -> None:
    _require_ipopt()
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="parameter")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
        {parameter: x},
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
    parameter.value = float(x.value)
    lower.solve(solver=cp.CLARABEL)
    assert lower.status in cp.settings.SOLUTION_PRESENT
    assert returned_y == pytest.approx(float(y.value), abs=2e-4)


def test_optimistic_lp_selects_upper_preferred_lower_optimizer() -> None:
    _require_ipopt()
    x = cp.Variable(name="x", bounds=[0.0, 1.0])
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="parameter")
    lower = cp.Problem(cp.Minimize(0.0 * y), [y >= parameter, y <= 1.0])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
        lower,
        {parameter: x},
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
