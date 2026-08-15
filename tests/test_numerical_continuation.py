"""Numerical coverage for restoration, multistart, and continuation failures."""

from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

import blvpy.continuation as continuation
from blvpy import BilevelProblem, BilevelResult, LowerProblem
from blvpy.errors import InitializationError

_IPOPT_OPTIONS = {
    "hessian_approximation": "limited-memory",
    "print_level": 0,
    "sb": "yes",
    "tol": 1e-9,
}


def _quadratic_response_model() -> tuple[BilevelProblem, cp.Variable, cp.Variable]:
    x = cp.Variable(name="x", bounds=[-3.0, 3.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    outer_objective = cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0))
    model = BilevelProblem(outer_objective, lower)
    return model, x, y


def _notes(error: BaseException) -> str:
    return "\n".join(getattr(error, "__notes__", ()))


@pytest.mark.parametrize(
    ("kind", "expected_status"),
    [("infeasible", "infeasible"), ("unbounded", "unbounded")],
)
def test_fixed_lower_initialization_reports_real_clarabel_failure(
    kind: str,
    expected_status: str,
) -> None:
    x = cp.Variable(name="x", bounds=[0.0, 1.0])
    y = cp.Variable(name="y")
    if kind == "infeasible":
        lower = LowerProblem(
            cp.Minimize(cp.square(y)),
            [y >= x + 1.0, y <= x],
            parameters=[x],
        )
    else:
        lower = LowerProblem(cp.Minimize(-y), [y >= x], parameters=[x])
    model = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower)

    with pytest.raises(InitializationError) as caught:
        model.solve(starts=1)

    assert str(caught.value) == "Automatic initialization failed. Please initialize variables: x."
    assert f"status '{expected_status}'" in _notes(caught.value)


@pytest.mark.ipopt
def test_real_feasibility_restoration_reaches_active_upper_constraint(monkeypatch) -> None:
    x = cp.Variable(name="x", bounds=[-3.0, 3.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
        outer_constraints=[x + y >= 1.5],
    )
    original = continuation._restore_feasibility
    observed: list[tuple[float, float]] = []

    def call_through(
        current,
        epsilon,
        solver,
        options,
        verbose,
        tolerance=1e-7,
    ) -> None:
        before = continuation.compute_residuals(current, epsilon).max_violation
        original(current, epsilon, solver, options, verbose, tolerance)
        after = continuation.compute_residuals(current, epsilon).max_violation
        observed.append((before, after))

    monkeypatch.setattr(continuation, "_restore_feasibility", call_through)

    result = model.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-4,
        starts=1,
        seed=23,
        solver_options=_IPOPT_OPTIONS,
    )

    assert observed
    assert observed[0][0] > 1.0
    assert observed[0][1] <= 1e-7
    assert result.succeeded
    assert result.residuals is not None
    assert result.residuals.max_violation <= 1e-7
    np.testing.assert_allclose([x.value, y.value], [0.75, 0.75], atol=2e-4)
    assert result.objective == pytest.approx(1.125, abs=5e-4)
    assert float(x.value + y.value) == pytest.approx(1.5, abs=2e-4)


def _double_well_model() -> tuple[BilevelProblem, cp.Variable, cp.Variable]:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    x.value = 1.0
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    outer = cp.Minimize(cp.square(cp.square(x) - 1.0) + 0.2 * x + 0.0 * y)
    return BilevelProblem(outer, lower), x, y


def _solve_double_well(
    monkeypatch,
    *,
    starts: int,
    seed: int,
) -> tuple[BilevelResult, float, tuple[float, ...]]:
    model, x, _ = _double_well_model()
    original = continuation._initialize_lower
    initializations: list[float] = []

    def call_through(current, *args, **kwargs) -> None:
        initializations.append(float(x.value))
        original(current, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(continuation, "_initialize_lower", call_through)
        result = model.solve(
            epsilon_initial=1e-3,
            epsilon_target=1e-3,
            starts=starts,
            seed=seed,
            solver_options=_IPOPT_OPTIONS,
        )
    return result, float(x.value), tuple(initializations)


@pytest.mark.ipopt
def test_seeded_real_multistart_escapes_inferior_local_minimum(monkeypatch) -> None:
    single, single_x, _ = _solve_double_well(monkeypatch, starts=1, seed=19)
    first, first_x, first_starts = _solve_double_well(monkeypatch, starts=5, seed=19)
    second, second_x, second_starts = _solve_double_well(monkeypatch, starts=5, seed=19)

    stationary = np.roots([4.0, 0.0, -4.0, 0.2])
    real_stationary = stationary[np.isclose(stationary.imag, 0.0)].real
    expected_global = min(
        real_stationary,
        key=lambda value: (value**2 - 1.0) ** 2 + 0.2 * value,
    )

    assert single.succeeded and first.succeeded and second.succeeded
    assert single_x > 0.9
    assert first_x == pytest.approx(expected_global, abs=2e-3)
    assert second_x == pytest.approx(expected_global, abs=2e-3)
    assert single.objective is not None and first.objective is not None
    assert single.objective - first.objective > 0.3
    assert first_starts == pytest.approx(second_starts)
    assert first_x == pytest.approx(second_x, abs=2e-6)
    assert first.objective == pytest.approx(second.objective, abs=2e-6)
    assert tuple(record.objective for record in first.starts) == pytest.approx(
        tuple(record.objective for record in second.starts),
        abs=2e-6,
    )


@pytest.mark.ipopt
def test_retry_uses_real_solves_after_one_injected_failure(monkeypatch) -> None:
    model, x, y = _quadratic_response_model()
    original = continuation._solve_one
    calls: list[float] = []
    injected = []

    def inject_one_failure(current, epsilon, solver, options, verbose):
        record = original(current, epsilon, solver, options, verbose)
        calls.append(epsilon)
        if np.isclose(epsilon, 1e-3) and not injected:
            injected.append(record)
            return replace(record, status="solver_error", message="injected retry signal")
        return record

    monkeypatch.setattr(continuation, "_solve_one", inject_one_failure)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-3,
        contraction=0.1,
        starts=1,
        seed=7,
        solver_options=_IPOPT_OPTIONS,
    )

    intermediate = np.sqrt(1e-5)
    assert injected and injected[0].residuals.max_violation <= 1e-7
    assert calls == pytest.approx([1e-1, 1e-2, 1e-3, intermediate, 1e-3])
    assert result.attempted_epsilon_history == pytest.approx(tuple(calls))
    assert result.epsilon_history == pytest.approx((1e-1, 1e-2, intermediate, 1e-3))
    assert result.succeeded
    assert result.final_epsilon == pytest.approx(1e-3)
    assert result.residuals is not None
    assert result.residuals.max_violation <= 1e-7
    expected = np.sqrt(1e-3) / 2.0
    assert float(x.value) == pytest.approx(expected, abs=5e-4)
    assert float(y.value) == pytest.approx(-expected, abs=5e-4)


@pytest.mark.ipopt
def test_infeasible_upper_constraints_report_restoration_reason() -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
        outer_constraints=[x >= 1.0, x <= 0.0],
    )

    with pytest.raises(InitializationError) as caught:
        model.solve(starts=1, solver_options=_IPOPT_OPTIONS)

    assert str(caught.value) == "Automatic initialization failed. Please initialize variables: x."
    notes = _notes(caught.value)
    assert "Feasibility restoration" in notes
    assert "max violation" in notes
