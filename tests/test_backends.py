"""Tests for the internal CVXPY solver invocation adapters."""

from __future__ import annotations

from unittest.mock import Mock

import cvxpy as cp
import pytest

from blvpy.backends import solve_conic, solve_dnlp
from blvpy.errors import SolverUnavailableError


def test_solve_conic_uses_regular_cvxpy_path() -> None:
    problem = Mock(spec=cp.Problem)

    result = solve_conic(
        problem,
        solver="CLARABEL",
        options={"max_iter": 25},
        verbose=True,
    )

    assert result is None
    problem.solve.assert_called_once_with(
        solver="CLARABEL",
        warm_start=True,
        verbose=True,
        max_iter=25,
    )
    assert "nlp" not in problem.solve.call_args.kwargs


def test_solve_dnlp_uses_nonlinear_cvxpy_path() -> None:
    problem = Mock(spec=cp.Problem)

    result = solve_dnlp(
        problem,
        solver="IPOPT",
        options={"max_iter": 50},
        verbose=False,
    )

    assert result is None
    problem.solve.assert_called_once_with(
        solver="IPOPT",
        nlp=True,
        warm_start=True,
        verbose=False,
        max_iter=50,
    )


@pytest.mark.parametrize(
    ("solver", "message"),
    [
        ("IPOPT", r"IPOPT.*native library.*required cyipopt"),
        ("CUSTOM", r"Requested solver 'CUSTOM'.*not installed or could not be loaded"),
    ],
)
def test_missing_solver_error_is_translated(solver: str, message: str) -> None:
    original = cp.SolverError(f"The solver {solver} is not installed.")
    problem = Mock(spec=cp.Problem)
    problem.solve.side_effect = original

    with pytest.raises(SolverUnavailableError, match=message) as raised:
        solve_dnlp(problem, solver=solver, options={}, verbose=False)

    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    "original",
    [
        ImportError("missing native module"),
        OSError("missing shared library"),
    ],
    ids=["import-error", "os-error"],
)
def test_native_loading_error_is_translated_with_detail(original: Exception) -> None:
    problem = Mock(spec=cp.Problem)
    problem.solve.side_effect = original

    with pytest.raises(SolverUnavailableError) as raised:
        solve_conic(problem, solver="CUSTOM", options={}, verbose=False)

    assert str(original) in str(raised.value)
    assert "Native loading error" in str(raised.value)
    assert raised.value.__cause__ is original


def test_unrelated_solver_error_is_preserved() -> None:
    original = cp.SolverError("Solver failed to converge.")
    problem = Mock(spec=cp.Problem)
    problem.solve.side_effect = original

    with pytest.raises(cp.SolverError) as raised:
        solve_dnlp(problem, solver="IPOPT", options={}, verbose=False)

    assert raised.value is original
