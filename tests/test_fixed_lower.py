"""Regression tests for reusable fixed-upper canonical lower solves."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy import BilevelProblem, LowerProblem
from blvpy.fixed_lower import solve_fixed_lower


def test_fixed_lower_solution_is_complete_immutable_and_does_not_touch_lifted_state() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(2, name="y")
    lower = LowerProblem(cp.Minimize(cp.sum_squares(y - cp.hstack([x, -x]))), parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.sum_squares(y)), lower)
    model.validate()
    linked_parameter = next(iter(model._parameter_links))
    x.value = 0.4
    linked_parameter.value = 0.4
    lifted = model._lifted_problem
    lifted.primal.value = np.full(lifted.primal.shape, 3.0)
    lifted.slack.value = np.full(lifted.slack.shape, 4.0)
    lifted.dual.value = np.full(lifted.dual.shape, 5.0)

    solution = solve_fixed_lower(model, cp.CLARABEL, {}, False)

    assert solution.status in cp.settings.SOLUTION_PRESENT
    np.testing.assert_allclose(solution.source_values[y.id], [0.4, -0.4], atol=1e-7)
    np.testing.assert_array_equal(lifted.primal.value, np.full(lifted.primal.shape, 3.0))
    np.testing.assert_array_equal(lifted.slack.value, np.full(lifted.slack.shape, 4.0))
    np.testing.assert_array_equal(lifted.dual.value, np.full(lifted.dual.shape, 5.0))
    with pytest.raises(ValueError):
        solution.primal[0] = 0.0
    with pytest.raises(ValueError):
        solution.source_values[y.id][0] = 0.0
    with pytest.raises(TypeError):
        solution.source_values[y.id] = np.zeros(2)  # type: ignore[index]


def test_fixed_lower_returns_power_cone_primal_dual_certificate() -> None:
    x = cp.Variable(name="x", bounds=[0.25, 1.5])
    y = cp.Variable(nonneg=True, name="y")
    lower = LowerProblem(
        cp.Minimize(cp.power(y, np.sqrt(2.0), approx=False)),
        [y >= x],
        parameters=[x],
    )
    model = BilevelProblem(cp.Minimize(cp.square(x - 0.7) + cp.square(y - 0.7)), lower)
    model.validate()
    linked_parameter = next(iter(model._parameter_links))
    x.value = 0.7
    linked_parameter.value = 0.7

    solution = solve_fixed_lower(
        model,
        cp.CLARABEL,
        {},
        False,
    )

    layout = model.canonicalize().cone_layout
    assert layout.power_3d
    assert float(solution.source_values[y.id]) == pytest.approx(0.7, abs=1e-7)
    assert layout.primal_distance(solution.slack) <= 1e-7
    assert layout.dual_distance(solution.dual) <= 1e-7
    assert abs(layout.complementarity(solution.slack, solution.dual)) <= 1e-7
