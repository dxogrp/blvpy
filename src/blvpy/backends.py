"""Internal CVXPY solver invocation adapters."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import cvxpy as cp

from .errors import SolverUnavailableError


def solve_conic(
    problem: cp.Problem,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> None:
    """Solve a conic problem through CVXPY's regular solve path."""

    _solve(problem, solver, options, solver_verbose, nlp=False)


def solve_dnlp(
    problem: cp.Problem,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> None:
    """Solve a DNLP problem through CVXPY's nonlinear solve path."""

    _solve(problem, solver, options, solver_verbose, nlp=True)


def _solve(
    problem: cp.Problem,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
    *,
    nlp: bool,
) -> None:
    solve_options = dict(options)
    if nlp and str(solver).upper() == "IPOPT" and not solver_verbose:
        solve_options.setdefault("print_level", 0)
        solve_options.setdefault("sb", "yes")
    try:
        if nlp:
            _solve_dnlp(problem, solver, solve_options, solver_verbose)
        else:
            problem.solve(
                solver=solver,
                warm_start=True,
                verbose=solver_verbose,
                **solve_options,
            )
    except cp.SolverError as error:
        if "not installed" in str(error).lower():
            raise _solver_unavailable_error(solver) from error
        raise
    except (ImportError, OSError) as error:
        raise _solver_unavailable_error(solver, detail=str(error)) from error


def _solve_dnlp(
    problem: cp.Problem,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> None:
    caught_warnings: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", UserWarning)
            problem.solve(
                solver=solver,
                nlp=True,
                warm_start=True,
                verbose=solver_verbose,
                **options,
            )
    except Exception as error:
        _replay_warnings(caught_warnings, primary_error=error)
        raise

    suppress = (
        caught_warnings[0]
        if (
            len(caught_warnings) == 1
            and caught_warnings[0].category is UserWarning
            and problem.status == cp.OPTIMAL_INACCURATE
            and problem.solver_stats is not None
        )
        else None
    )
    _replay_warnings(caught_warnings, suppress=suppress)


def _replay_warnings(
    caught_warnings: list[warnings.WarningMessage],
    *,
    suppress: warnings.WarningMessage | None = None,
    primary_error: Exception | None = None,
) -> None:
    for warning in caught_warnings:
        if warning is suppress:
            continue
        try:
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
                source=warning.source,
            )
        except Exception as replay_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Captured warning before solver failure: {type(replay_error).__name__}: {replay_error}"
            )


def _solver_unavailable_error(solver: str, *, detail: str | None = None) -> SolverUnavailableError:
    if str(solver).upper() == "IPOPT":
        message = (
            "IPOPT is not available. Install its native library, then reinstall "
            "BLVPY so its required cyipopt binding can be built."
        )
    else:
        message = f"Requested solver {solver!r} is not installed or could not be loaded by CVXPY."
    if detail:
        message = f"{message} Native loading error: {detail}"
    return SolverUnavailableError(message)
