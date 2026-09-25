"""Tests for the internal CVXPY solver invocation adapters."""

from __future__ import annotations

import warnings
from unittest.mock import Mock

import cvxpy as cp
import pytest
from cvxpy.reductions.solution import Solution

from blvpy.backends import solve_conic, solve_dnlp
from blvpy.errors import SolverUnavailableError


def test_solve_conic_uses_regular_cvxpy_path() -> None:
    problem = Mock(spec=cp.Problem)

    result = solve_conic(
        problem,
        solver="CLARABEL",
        options={"max_iter": 25},
        solver_verbose=True,
    )

    assert result is None
    problem.solve.assert_called_once_with(
        solver="CLARABEL",
        warm_start=True,
        verbose=True,
        max_iter=25,
    )
    assert "nlp" not in problem.solve.call_args.kwargs


@pytest.mark.parametrize(
    "solver",
    [
        cp.IPOPT,
        cp.KNITRO,
        cp.UNO,
        cp.COPT,
        "knitro_ipm",
        "knitro_sqp",
        "knitro_alm",
        "uno_ipm",
        "uno_sqp",
    ],
)
def test_solve_dnlp_uses_selected_nonlinear_cvxpy_path(solver: str) -> None:
    problem = Mock(spec=cp.Problem)

    result = solve_dnlp(
        problem,
        solver=solver,
        options={"max_iter": 50},
        solver_verbose=False,
    )

    assert result is None
    expected = {
        "solver": solver,
        "nlp": True,
        "warm_start": True,
        "verbose": False,
        "max_iter": 50,
    }
    if solver == cp.IPOPT:
        expected.update(print_level=0, sb="yes")
    problem.solve.assert_called_once_with(**expected)


def test_solve_dnlp_does_not_add_quiet_options_when_solver_is_verbose() -> None:
    problem = Mock(spec=cp.Problem)

    solve_dnlp(problem, solver="IPOPT", options={}, solver_verbose=True)

    problem.solve.assert_called_once_with(
        solver="IPOPT",
        nlp=True,
        warm_start=True,
        verbose=True,
    )


def test_solve_dnlp_preserves_explicit_ipopt_output_options() -> None:
    problem = Mock(spec=cp.Problem)
    options = {"print_level": 4, "sb": "no"}

    solve_dnlp(problem, solver="IPOPT", options=options, solver_verbose=False)

    problem.solve.assert_called_once_with(
        solver="IPOPT",
        nlp=True,
        warm_start=True,
        verbose=False,
        print_level=4,
        sb="no",
    )
    assert options == {"print_level": 4, "sb": "no"}


def test_solve_dnlp_unpacks_an_inaccurate_cvxpy_result_under_werror() -> None:
    problem = cp.Problem(cp.Minimize(0.0))
    solution = Solution(cp.OPTIMAL_INACCURATE, 0.0, {}, {}, {})
    chain = Mock()
    chain.invert.return_value = solution
    chain.solver.name.return_value = "TEST"
    problem.solve = lambda **_options: problem.unpack_results(solution, chain, [])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_dnlp(problem, solver="TEST", options={}, solver_verbose=False)

    assert problem.status == cp.OPTIMAL_INACCURATE
    assert problem.solver_stats.solver_name == "TEST"


def test_solve_dnlp_replays_an_additional_user_warning_under_werror() -> None:
    problem = cp.Problem(cp.Minimize(0.0))
    solution = Solution(cp.OPTIMAL_INACCURATE, 0.0, {}, {}, {})
    chain = Mock()
    chain.invert.return_value = solution
    chain.solver.name.return_value = "TEST"

    def solve(**_options: object) -> None:
        warnings.warn("additional solver warning", UserWarning, stacklevel=1)
        problem.unpack_results(solution, chain, [])

    problem.solve = solve

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="additional solver warning"):
            solve_dnlp(problem, solver="TEST", options={}, solver_verbose=False)

    assert problem.status == cp.OPTIMAL_INACCURATE
    assert problem.solver_stats.solver_name == "TEST"


@pytest.mark.parametrize(
    "solver",
    [cp.KNITRO, cp.UNO, cp.COPT, "knitro_ipm", "knitro_sqp", "knitro_alm", "uno_ipm", "uno_sqp"],
)
def test_quiet_ipopt_defaults_do_not_apply_to_other_dnlp_solvers(solver: str) -> None:
    problem = Mock(spec=cp.Problem)
    options = {"backend_option": "preserved"}

    solve_dnlp(problem, solver=solver, options=options, solver_verbose=False)

    problem.solve.assert_called_once_with(
        solver=solver,
        nlp=True,
        warm_start=True,
        verbose=False,
        backend_option="preserved",
    )
    assert options == {"backend_option": "preserved"}


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
        solve_dnlp(problem, solver=solver, options={}, solver_verbose=False)

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
        solve_conic(problem, solver="CUSTOM", options={}, solver_verbose=False)

    assert str(original) in str(raised.value)
    assert "Native loading error" in str(raised.value)
    assert raised.value.__cause__ is original


def test_unrelated_solver_error_is_preserved() -> None:
    original = cp.SolverError("Solver failed to converge.")
    problem = Mock(spec=cp.Problem)
    problem.solve.side_effect = original

    with pytest.raises(cp.SolverError) as raised:
        solve_dnlp(problem, solver="IPOPT", options={}, solver_verbose=False)

    assert raised.value is original
