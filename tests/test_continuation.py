"""Solver-independent initialization and continuation tests."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

import blvpy.continuation as continuation
from blvpy import BilevelProblem
from blvpy.continuation import compute_residuals, sample_upper_starts
from blvpy.errors import InitializationError, SolverUnavailableError
from blvpy.result import IterationRecord, Residuals


def _quadratic_bilevel(*, bounded: bool = True):
    attributes = {"bounds": [-2.0, 2.0]} if bounded else {}
    x = cp.Variable(name="x", **attributes)
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="lower_x")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
        {parameter: x},
        [x + y <= 3.0],
    )
    return model, x, y, parameter


def test_bounded_upper_sampling_is_reproducible_and_uses_explicit_first() -> None:
    model, x, _, _ = _quadratic_bilevel()
    x.value = 0.75

    first = sample_upper_starts(model, 5, np.random.default_rng(19))
    second = sample_upper_starts(model, 5, np.random.default_rng(19))

    assert first[0][x] == pytest.approx(0.75)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left[x], right[x])
        assert -2.0 <= float(left[x]) <= 2.0
    assert any(float(sample[x]) != pytest.approx(0.75) for sample in first[1:])


def test_unbounded_upper_requires_value_or_sample_bounds() -> None:
    model, x, _, _ = _quadratic_bilevel(bounded=False)

    with pytest.raises(InitializationError, match="unbounded component"):
        sample_upper_starts(model, 1, np.random.default_rng(0))

    x.value = -3.25
    repeated = sample_upper_starts(model, 3, np.random.default_rng(0))
    assert [float(sample[x]) for sample in repeated] == pytest.approx([-3.25] * 3)


def test_sample_bounds_supports_unbounded_vector_variable() -> None:
    x = cp.Variable(3, name="x")
    x.sample_bounds = (np.array([-3.0, -2.0, -1.0]), np.array([-2.0, 0.0, 4.0]))
    y = cp.Variable(name="y")
    parameter = cp.Parameter(3, name="lower_x")
    lower = cp.Problem(cp.Minimize(cp.square(y) + cp.sum_squares(parameter)))
    model = BilevelProblem(cp.Minimize(cp.sum_squares(x) + cp.square(y)), lower, {parameter: x})

    starts = sample_upper_starts(model, 20, np.random.default_rng(4))
    values = np.vstack([sample[x] for sample in starts])
    assert np.all(values >= np.array([-3.0, -2.0, -1.0]))
    assert np.all(values <= np.array([-2.0, 0.0, 4.0]))


def test_sampling_tracks_bounds_per_variable_when_bound_types_are_mixed() -> None:
    bounded = cp.Variable(name="bounded", bounds=[-2.0, -1.0])
    explicit_only = cp.Variable(name="explicit_only")
    explicit_only.value = 7.5
    source = cp.Variable(name="source")
    parameter = cp.Parameter(name="parameter")
    lower = cp.Problem(cp.Minimize(cp.square(source - parameter)))
    model = BilevelProblem(
        cp.Minimize(cp.square(bounded) + cp.square(explicit_only) + cp.square(source)),
        lower,
        {parameter: bounded},
    )

    samples = sample_upper_starts(model, 8, np.random.default_rng(18))

    assert all(-2.0 <= float(sample[bounded]) <= -1.0 for sample in samples)
    assert [float(sample[explicit_only]) for sample in samples] == pytest.approx([7.5] * 8)


def test_default_solve_reports_actionable_missing_ipopt_error(monkeypatch) -> None:
    model, _, _, _ = _quadratic_bilevel()
    installed = [solver for solver in cp.installed_solvers() if str(solver).upper() != "IPOPT"]
    monkeypatch.setattr(cp, "installed_solvers", lambda: installed)

    with pytest.raises(SolverUnavailableError, match=r"IPOPT.*required cyipopt"):
        model.solve()


def test_solve_argument_validation_precedes_numerical_backends() -> None:
    model, _, _, _ = _quadratic_bilevel()

    with pytest.raises(ValueError, match="epsilon_target cannot exceed"):
        model.solve(epsilon_initial=1e-3, epsilon_target=1e-2)
    with pytest.raises(ValueError, match="contraction"):
        model.solve(contraction=1.0)
    with pytest.raises(ValueError, match="starts"):
        model.solve(starts=0)


def test_compute_residuals_matches_independent_canonical_calculation() -> None:
    model, x, y, parameter = _quadratic_bilevel()
    lifted = model.lifted_problem
    canonical = model.canonicalize()
    x.value = 0.4
    parameter.value = x.value
    data = canonical.apply_numeric({parameter: x.value})

    primal = cp.Variable(canonical.canonical_size)
    slack = cp.Variable(canonical.constraint_size)
    equality = data.A @ primal + slack == data.b
    lower = cp.Problem(
        cp.Minimize(data.c @ primal),
        [equality, *canonical.cone_layout.primal_constraints(slack)],
    )
    lower.solve(solver=cp.CLARABEL)
    assert lower.status in cp.settings.SOLUTION_PRESENT

    lifted.primal.value = primal.value
    lifted.slack.value = slack.value
    lifted.dual.value = equality.dual_value
    y.value = canonical.recover_numeric(primal.value)[y.id]
    epsilon = 1e-5
    actual = compute_residuals(model, epsilon)

    u = np.asarray(primal.value).reshape(-1)
    s = np.asarray(slack.value).reshape(-1)
    dual = np.asarray(equality.dual_value).reshape(-1)
    expected_primal = np.linalg.norm(data.A @ u + s - data.b)
    expected_dual = np.linalg.norm(data.A.T @ dual + data.c)
    expected_complementarity = float(s @ dual)
    assert actual.primal_equality == pytest.approx(expected_primal)
    assert actual.dual_equality == pytest.approx(expected_dual)
    assert actual.recovery == pytest.approx(0.0)
    assert actual.upper_constraints == pytest.approx(0.0)
    assert actual.primal_cone == pytest.approx(canonical.cone_layout.primal_distance(s))
    assert actual.dual_cone == pytest.approx(canonical.cone_layout.dual_distance(dual))
    assert actual.complementarity == pytest.approx(expected_complementarity)
    assert actual.gap_violation == pytest.approx(max(expected_complementarity - epsilon, 0.0))


def test_fixed_upper_initialization_recovers_direct_lower_solution() -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(2, name="y")
    parameter = cp.Parameter(name="parameter")
    lower = cp.Problem(
        cp.Minimize(cp.sum_squares(y - cp.hstack([parameter, -parameter])))
    )
    model = BilevelProblem(cp.Minimize(cp.square(x) + cp.sum_squares(y)), lower, {parameter: x})
    x.value = 0.65

    continuation._initialize_lower(model, cp.CLARABEL, {}, False)
    initialized = np.array(y.value, copy=True)

    parameter.value = x.value
    lower.solve(solver=cp.CLARABEL)
    assert lower.status in cp.settings.SOLUTION_PRESENT
    np.testing.assert_allclose(initialized, y.value, atol=2e-5)


def test_compute_residuals_reports_each_independent_failure() -> None:
    model, x, y, parameter = _quadratic_bilevel()
    lifted = model.lifted_problem
    canonical = model.canonicalize()
    x.value = 2.0
    parameter.value = x.value
    lifted.primal.value = np.zeros(canonical.canonical_size)
    lifted.slack.value = -np.ones(canonical.constraint_size)
    lifted.dual.value = np.ones(canonical.constraint_size)
    y.value = 10.0

    residuals = compute_residuals(model, epsilon=0.0)

    assert residuals.primal_equality > 0
    assert residuals.dual_equality > 0
    assert residuals.recovery > 0
    assert residuals.upper_constraints > 0
    assert residuals.primal_cone > 0
    assert residuals.complementarity < 0
    assert residuals.gap_violation == 0


def test_failed_step_inserts_intermediate_epsilon_then_retries(monkeypatch) -> None:
    model, x, y, _ = _quadratic_bilevel()
    calls: list[float] = []
    failed_target_once = False
    zero = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(continuation, "_require_solver", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)

    def fake_initialize(current, *args, **kwargs) -> None:
        lifted = current.lifted_problem
        y.value = float(x.value)
        lifted.primal.value = np.zeros(lifted.primal.size)
        lifted.slack.value = np.zeros(lifted.slack.size)
        lifted.dual.value = np.zeros(lifted.dual.size)

    def fake_solve(current, epsilon, solver, options, verbose):
        nonlocal failed_target_once
        calls.append(epsilon)
        if epsilon == pytest.approx(1e-2) and not failed_target_once:
            failed_target_once = True
            return IterationRecord(epsilon, "solver_error", None, zero, solver_name=str(solver))
        current.lifted_problem.epsilon.value = epsilon
        return IterationRecord(
            epsilon,
            cp.OPTIMAL,
            float(current.outer_objective.value),
            zero,
            solver_name=str(solver),
        )

    monkeypatch.setattr(continuation, "_initialize_lower", fake_initialize)
    monkeypatch.setattr(continuation, "_solve_one", fake_solve)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: zero)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        starts=1,
        seed=7,
        solver="MOCK_NLP",
        restoration=False,
    )

    assert result.status == cp.OPTIMAL
    # The original failed target is recorded, then an intermediate value is
    # inserted and the target is retried. Floating contraction may also cause
    # the pre-existing schedule to contain a value numerically equal to target.
    assert calls[0] == pytest.approx(1e-1)
    assert calls[1] == pytest.approx(1e-2)
    assert any(value == pytest.approx(np.sqrt(1e-3)) for value in calls)
    assert calls[-1] == pytest.approx(1e-2)
    assert len(calls) == 4
    assert result.epsilon_history == pytest.approx((1e-1, np.sqrt(1e-3), 1e-2))
    assert result.attempted_epsilon_history == pytest.approx(tuple(calls))
    assert result.final_epsilon == pytest.approx(1e-2)
    assert result.starts[0].status == cp.OPTIMAL


def test_retry_budget_stops_repeated_bisection_and_returns_consistent_point(
    monkeypatch,
) -> None:
    model, x, y, _ = _quadratic_bilevel()
    zero = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    calls: list[float] = []

    monkeypatch.setattr(continuation, "_require_solver", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "_compile_probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuation, "compute_residuals", lambda *args, **kwargs: zero)

    def fake_initialize(current, *args, **kwargs) -> None:
        lifted = current.lifted_problem
        y.value = float(x.value)
        lifted.primal.value = np.zeros(lifted.primal.size)
        lifted.slack.value = np.zeros(lifted.slack.size)
        lifted.dual.value = np.zeros(lifted.dual.size)

    def fake_solve(current, epsilon, solver, options, verbose):
        calls.append(epsilon)
        status = cp.OPTIMAL if epsilon > 2e-2 else "solver_error"
        current.lifted_problem.epsilon.value = epsilon
        return IterationRecord(epsilon, status, 0.0 if status == cp.OPTIMAL else None, zero)

    monkeypatch.setattr(continuation, "_initialize_lower", fake_initialize)
    monkeypatch.setattr(continuation, "_solve_one", fake_solve)

    result = model.solve(
        epsilon_initial=1e-1,
        epsilon_target=1e-2,
        starts=1,
        solver="MOCK_NLP",
        restoration=False,
        max_retries=2,
    )

    assert result.status == "continuation_failed"
    assert len(calls) == 5
    assert result.final_epsilon == pytest.approx(np.sqrt(1e-3))
    assert result.residuals == zero
