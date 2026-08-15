"""Solver-independent tests for complete gap diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import cvxpy as cp
import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from blvpy import BilevelProblem, BilevelResult, GapDiagnostics, LowerProblem
from blvpy.diagnostics import _from_canonical
from blvpy.errors import SolveError, SolverUnavailableError


def _scalar_lp() -> tuple[
    BilevelProblem,
    cp.Variable,
    cp.Variable,
    cp.Parameter,
    cp.Parameter,
]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    fixed_weight = cp.Parameter(nonneg=True, value=1.0, name="fixed_weight")
    lower = LowerProblem(
        cp.Minimize(fixed_weight * y + 3.0 * x - 2.0),
        [y >= x],
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
    )
    model.canonicalize()
    linked_parameter = next(iter(model._parameter_links))
    return model, x, y, fixed_weight, linked_parameter


def _complete_result(
    x: cp.Variable,
    y: cp.Variable,
    *,
    status: str = "optimal",
    primal: ArrayLike | None = (0.65,),
    slack: ArrayLike | None = (0.25,),
    dual: ArrayLike | None = (1.0,),
    variable_values: dict[cp.Variable, ArrayLike] | None = None,
) -> BilevelResult:
    return BilevelResult(
        status=status,
        objective=0.4**2 + 0.65**2,
        variable_values={x: np.array(0.4), y: np.array(0.65)} if variable_values is None else variable_values,
        canonical_primal=primal,
        slack=slack,
        dual=dual,
    )


def _snapshot(value: ArrayLike | None) -> NDArray[np.float64] | None:
    return None if value is None else np.array(value, dtype=float, copy=True)


def _assert_value(actual: ArrayLike | None, expected: NDArray[np.float64] | None) -> None:
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, expected)


@contextmanager
def _non_result_state(
    model: BilevelProblem,
    x: cp.Variable,
    y: cp.Variable,
    fixed_weight: cp.Parameter,
    linked_parameter: cp.Parameter,
) -> Iterator[dict[str, NDArray[np.float64] | None]]:
    lifted = model._lifted_problem
    x.value = 1.2
    y.value = 1.3
    fixed_weight.value = 2.0
    linked_parameter.value = 0.9
    lifted.primal.value = np.full(lifted.primal.shape, 3.0)
    lifted.slack.value = np.full(lifted.slack.shape, 4.0)
    lifted.dual.value = np.full(lifted.dual.shape, 5.0)
    lifted.epsilon.value = 0.03
    state = {
        "x": _snapshot(x.value),
        "y": _snapshot(y.value),
        "fixed_weight": _snapshot(fixed_weight.value),
        "linked_parameter": _snapshot(linked_parameter.value),
        "primal": _snapshot(lifted.primal.value),
        "slack": _snapshot(lifted.slack.value),
        "dual": _snapshot(lifted.dual.value),
        "epsilon": _snapshot(lifted.epsilon.value),
    }
    try:
        yield state
    finally:
        _assert_value(x.value, state["x"])
        _assert_value(y.value, state["y"])
        _assert_value(fixed_weight.value, state["fixed_weight"])
        _assert_value(linked_parameter.value, state["linked_parameter"])
        _assert_value(lifted.primal.value, state["primal"])
        _assert_value(lifted.slack.value, state["slack"])
        _assert_value(lifted.dual.value, state["dual"])
        _assert_value(lifted.epsilon.value, state["epsilon"])


def test_private_canonical_helper_satisfies_inexact_gap_identity() -> None:
    a = np.array([[1.5, -0.5], [0.25, 2.0], [-1.0, 1.0]])
    b = np.array([1.0, -2.0, 0.5])
    c = np.array([0.75, -1.25])
    primal = np.array([0.4, -0.7])
    slack = np.array([0.2, 0.3, -0.1])
    dual = np.array([0.8, -0.2, 1.1])
    primal_residual = a @ primal + slack - b
    dual_residual = a.T @ dual + c

    diagnostics = _from_canonical(
        c=c,
        b=b,
        primal=primal,
        dual=dual,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        complementarity=float(slack @ dual),
        source_gap=0.125,
    )

    assert diagnostics.primal_objective == pytest.approx(float(c @ primal))
    assert diagnostics.dual_objective == pytest.approx(float(-(b @ dual)))
    assert diagnostics.complementarity == pytest.approx(float(slack @ dual))
    assert diagnostics.dual_residual_term == pytest.approx(float(primal @ dual_residual))
    assert diagnostics.primal_residual_term == pytest.approx(float(dual @ primal_residual))
    assert diagnostics.source_gap == pytest.approx(0.125)
    assert diagnostics.identity_error == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("status", ["optimal", "continuation_failed"])
def test_gap_diagnostics_computes_complete_hand_derived_lp_terms_and_restores_state(
    status: str,
) -> None:
    model, x, y, fixed_weight, linked_parameter = _scalar_lp()
    result = _complete_result(x, y, status=status)
    result_snapshot = {variable: np.array(value, copy=True) for variable, value in result.variable_values.items()}

    with _non_result_state(model, x, y, fixed_weight, linked_parameter):
        diagnostics = model.gap_diagnostics(result)

    assert isinstance(diagnostics, GapDiagnostics)
    assert diagnostics.primal_objective == pytest.approx(0.65, abs=1e-12)
    assert diagnostics.dual_objective == pytest.approx(0.4, abs=1e-12)
    assert diagnostics.complementarity == pytest.approx(0.25, abs=1e-12)
    assert diagnostics.dual_residual_term == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.primal_residual_term == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.normalized_gap == pytest.approx(0.25, abs=1e-12)
    assert diagnostics.inexact_identity_rhs == pytest.approx(0.25, abs=1e-12)
    assert diagnostics.identity_error == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.source_gap == pytest.approx(0.25, abs=1e-8)
    for variable, value in result_snapshot.items():
        np.testing.assert_array_equal(result.variable_values[variable], value)


def test_gap_diagnostics_invokes_exactly_one_additional_quiet_clarabel_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, _, _ = _scalar_lp()
    result = _complete_result(x, y)
    from blvpy import diagnostics as diagnostics_module

    real_solve = diagnostics_module.solve_conic
    calls: list[tuple[str, dict[object, object], bool]] = []

    def recording_solve(problem, solver, options, solver_verbose):
        assert problem is not model._cvxpy_lower_problem
        calls.append((solver, dict(options), solver_verbose))
        return real_solve(problem, solver, options, solver_verbose)

    monkeypatch.setattr(diagnostics_module, "solve_conic", recording_solve)

    assert calls == []
    model.gap_diagnostics(result)

    assert calls == [(cp.CLARABEL, {}, False)]


def test_gap_diagnostics_forwards_selected_solver_copied_options_and_verbosity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, _, _ = _scalar_lp()
    result = _complete_result(x, y)
    from blvpy import diagnostics as diagnostics_module

    original_options = {"eps": 1e-7, "max_iters": 250}
    calls: list[tuple[str, dict[str, float | int], bool]] = []

    def recording_solve(problem, solver, options, solver_verbose):
        assert solver == cp.SCS
        assert options == original_options
        assert options is not original_options
        calls.append((solver, dict(options), solver_verbose))
        diagnostics_module.cp.Problem.solve(
            problem,
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=False,
        )

    monkeypatch.setattr(diagnostics_module, "solve_conic", recording_solve)

    diagnostics = model.gap_diagnostics(
        result,
        solver=cp.SCS,
        solver_options=original_options,
        solver_verbose=True,
    )

    assert calls == [(cp.SCS, original_options, True)]
    assert original_options == {"eps": 1e-7, "max_iters": 250}
    assert diagnostics.source_gap == pytest.approx(0.25, abs=1e-8)


def test_gap_diagnostics_solves_reference_problem_with_scs() -> None:
    model, x, y, _, _ = _scalar_lp()
    result = _complete_result(x, y)

    diagnostics = model.gap_diagnostics(
        result,
        solver=cp.SCS,
        solver_options={"eps": 1e-8, "max_iters": 10_000},
    )

    assert diagnostics.source_gap == pytest.approx(0.25, abs=5e-7)


@pytest.mark.parametrize("solver_verbose", [0, 1, "false", None])
def test_gap_diagnostics_rejects_non_boolean_solver_verbose(solver_verbose: object) -> None:
    model, x, y, _, _ = _scalar_lp()
    result = _complete_result(x, y)

    with pytest.raises(ValueError, match="solver_verbose must be boolean"):
        model.gap_diagnostics(result, solver_verbose=solver_verbose)  # type: ignore[arg-type]


def test_gap_diagnostics_requires_an_explicit_conic_solver() -> None:
    model, x, y, _, _ = _scalar_lp()
    result = _complete_result(x, y)

    with pytest.raises(ValueError, match="solver must name a CVXPY conic backend"):
        model.gap_diagnostics(result, solver=None)  # type: ignore[arg-type]


def test_gap_diagnostics_preserves_a_small_signed_negative_source_gap() -> None:
    model, x, y, _, _ = _scalar_lp()
    below_reference = 0.4 - 1e-8
    result = _complete_result(
        x,
        y,
        primal=(below_reference,),
        slack=(-1e-8,),
        variable_values={x: np.array(0.4), y: np.array(below_reference)},
    )

    diagnostics = model.gap_diagnostics(result)

    assert diagnostics.source_gap == pytest.approx(-1e-8, abs=2e-9)


def test_gap_diagnostics_rejects_non_result() -> None:
    model, _, _, _, _ = _scalar_lp()

    with pytest.raises(TypeError, match="BilevelResult"):
        model.gap_diagnostics(object())  # type: ignore[arg-type]


def test_gap_diagnostics_rejects_result_from_another_problem() -> None:
    first, first_x, first_y, _, _ = _scalar_lp()
    second, _, _, _, _ = _scalar_lp()
    result = _complete_result(first_x, first_y)

    with pytest.raises(ValueError, match="variable_values|source variables"):
        second.gap_diagnostics(result)

    assert first is not second


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("primal", "canonical_primal"),
        ("slack", "slack"),
        ("dual", "dual"),
    ],
)
def test_gap_diagnostics_rejects_missing_canonical_arrays(missing: str, message: str) -> None:
    model, x, y, _, _ = _scalar_lp()
    values: dict[str, ArrayLike | None] = {
        "primal": (0.65,),
        "slack": (0.25,),
        "dual": (1.0,),
    }
    values[missing] = None
    result = _complete_result(
        x,
        y,
        primal=values["primal"],
        slack=values["slack"],
        dual=values["dual"],
    )

    with pytest.raises(ValueError, match=message):
        model.gap_diagnostics(result)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("primal", "canonical_primal"),
        ("slack", "slack"),
        ("dual", "dual"),
    ],
)
def test_gap_diagnostics_rejects_incorrect_canonical_dimensions(field: str, message: str) -> None:
    model, x, y, _, _ = _scalar_lp()
    values: dict[str, ArrayLike] = {
        "primal": (0.65,),
        "slack": (0.25,),
        "dual": (1.0,),
    }
    values[field] = (1.0, 2.0)
    result = _complete_result(
        x,
        y,
        primal=values["primal"],
        slack=values["slack"],
        dual=values["dual"],
    )

    with pytest.raises(ValueError, match=message):
        model.gap_diagnostics(result)


def test_gap_diagnostics_propagates_solver_unavailable_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, _, _ = _scalar_lp()
    result = _complete_result(x, y)
    unavailable = SolverUnavailableError("Clarabel unavailable")

    def fail(*args, **kwargs):
        raise unavailable

    monkeypatch.setattr("blvpy.diagnostics.solve_conic", fail)

    with pytest.raises(SolverUnavailableError) as raised:
        model.gap_diagnostics(result)

    assert raised.value is unavailable


def test_gap_diagnostics_wraps_reference_solver_error_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, fixed_weight, linked_parameter = _scalar_lp()
    result = _complete_result(x, y)
    options = {"custom_tolerance": 1e-9}

    def fail(problem, solver, received_options, solver_verbose):
        assert problem is not model._cvxpy_lower_problem
        assert solver == "CUSTOM"
        assert received_options == options
        assert received_options is not options
        assert solver_verbose is True
        raise cp.SolverError("reference numerical failure")

    monkeypatch.setattr("blvpy.diagnostics.solve_conic", fail)

    with _non_result_state(model, x, y, fixed_weight, linked_parameter):
        with pytest.raises(SolveError, match="reference numerical failure"):
            model.gap_diagnostics(
                result,
                solver="CUSTOM",
                solver_options=options,
                solver_verbose=True,
            )

    assert options == {"custom_tolerance": 1e-9}


def test_gap_diagnostics_reports_reference_failure_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, fixed_weight, linked_parameter = _scalar_lp()
    result = _complete_result(x, y)

    def report_infeasible(problem, solver, options, solver_verbose):
        assert solver == cp.CLARABEL
        assert options == {}
        assert solver_verbose is False
        for variable in problem.variables():
            variable.save_value(np.full(variable.shape, 123.0))
        for parameter in problem.parameters():
            parameter.save_value(np.full(parameter.shape, 456.0))
        problem._status = cp.INFEASIBLE
        problem._value = np.inf

    monkeypatch.setattr("blvpy.diagnostics.solve_conic", report_infeasible)

    with _non_result_state(model, x, y, fixed_weight, linked_parameter):
        with pytest.raises(SolveError, match="fixed-upper lower|reference|infeasible"):
            model.gap_diagnostics(result)
