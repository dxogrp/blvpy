"""Solver-independent tests for lifted residual and result diagnostics."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy import BilevelProblem, LowerProblem
from blvpy.continuation import compute_residuals
from blvpy.errors import InitializationError
from blvpy.result import Residuals


def _linear_bilevel() -> tuple[BilevelProblem, cp.Variable, cp.Variable]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(y), [y >= x], parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
        outer_constraints=[x <= 1.0, x >= -4.0],
    )
    return model, x, y


def test_compute_residuals_reports_dual_cone_and_exact_upper_violation() -> None:
    model, x, y = _linear_bilevel()
    canonical = model.canonicalize()
    lifted = model.lifted_problem
    assert canonical.cone_layout.nonnegative == 1
    assert canonical.constraint_size == 1

    x.value = 3.0
    y.value = 0.0
    lifted.primal.value = np.zeros(canonical.canonical_size)
    lifted.slack.value = np.zeros(canonical.constraint_size)
    lifted.dual.value = np.array([-2.0])

    residuals = compute_residuals(model, epsilon=0.0)

    assert residuals.upper_constraints == pytest.approx(2.0)
    assert residuals.dual_cone == pytest.approx(2.0)
    assert residuals.primal_cone == pytest.approx(0.0)
    assert residuals.complementarity == pytest.approx(0.0)
    assert residuals.gap_violation == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("primal", "canonical primal"),
        ("slack", "canonical slack"),
        ("dual", "canonical dual"),
    ],
)
def test_compute_residuals_rejects_missing_canonical_values(missing: str, message: str) -> None:
    model, x, _ = _linear_bilevel()
    lifted = model.lifted_problem
    x.value = 0.0
    values = {
        "primal": np.zeros(lifted.primal.size),
        "slack": np.zeros(lifted.slack.size),
        "dual": np.zeros(lifted.dual.size),
    }
    values[missing] = None
    lifted.primal.value = values["primal"]
    lifted.slack.value = values["slack"]
    lifted.dual.value = values["dual"]

    with pytest.raises(InitializationError, match=rf"The {message} has no numeric value"):
        compute_residuals(model, epsilon=0.0)


def test_compute_residuals_rejects_missing_linked_value() -> None:
    model, _, _ = _linear_bilevel()

    with pytest.raises(InitializationError, match="Linked upper variables do not all have numeric values"):
        compute_residuals(model, epsilon=0.0)


def test_missing_lower_source_value_produces_infinite_recovery_residual() -> None:
    model, x, y = _linear_bilevel()
    lifted = model.lifted_problem
    x.value = 0.0
    y.value = None
    lifted.primal.value = np.zeros(lifted.primal.size)
    lifted.slack.value = np.zeros(lifted.slack.size)
    lifted.dual.value = np.ones(lifted.dual.size)

    residuals = compute_residuals(model, epsilon=0.0)

    assert np.isinf(residuals.recovery)
    assert not residuals.is_feasible(1e-7)


def test_is_feasible_uses_separate_gap_tolerance_and_ignores_source_gap() -> None:
    tolerance = 1e-5
    residuals = Residuals(
        primal_equality=tolerance,
        dual_equality=0.0,
        recovery=0.0,
        upper_constraints=0.0,
        primal_cone=0.0,
        dual_cone=0.0,
        complementarity=3e-5,
        gap_violation=2e-5,
        source_gap=1e6,
    )

    assert residuals.max_feasibility == tolerance
    assert residuals.max_violation == 2e-5
    assert not residuals.is_feasible(tolerance)
    assert residuals.is_feasible(tolerance, gap_tolerance=2e-5)
    assert not residuals.is_feasible(tolerance, gap_tolerance=2e-5 - 1e-12)

    with_different_source_gap = Residuals(
        **{**residuals.as_dict(), "source_gap": -1e6},
    )
    assert with_different_source_gap.is_feasible(tolerance, gap_tolerance=2e-5)


def test_nonfinite_residuals_are_never_feasible() -> None:
    residuals = Residuals(
        primal_equality=float("inf"),
        dual_equality=0.0,
        recovery=0.0,
        upper_constraints=0.0,
        primal_cone=0.0,
        dual_cone=0.0,
        complementarity=float("inf"),
        gap_violation=float("inf"),
        source_gap=float("inf"),
    )

    assert np.isinf(residuals.max_feasibility)
    assert np.isinf(residuals.max_violation)
    assert not residuals.is_feasible(1e10, gap_tolerance=1e10)
    assert np.isinf(residuals.as_dict()["source_gap"])
    with pytest.raises(ValueError, match="finite"):
        residuals.is_feasible(float("inf"))


@pytest.mark.parametrize(
    "field",
    [
        "primal_equality",
        "dual_equality",
        "recovery",
        "upper_constraints",
        "primal_cone",
        "dual_cone",
        "gap_violation",
    ],
)
def test_residuals_reject_nan_violations(field: str) -> None:
    values = {
        "primal_equality": 0.0,
        "dual_equality": 0.0,
        "recovery": 0.0,
        "upper_constraints": 0.0,
        "primal_cone": 0.0,
        "dual_cone": 0.0,
        "complementarity": 0.0,
        "gap_violation": 0.0,
    }
    values[field] = float("nan")

    with pytest.raises(ValueError, match="nonnegative and not NaN"):
        Residuals(**values)
