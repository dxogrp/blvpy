"""Internal CVXPY solver invocation adapters."""

from __future__ import annotations

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
            problem.solve(
                solver=solver,
                nlp=True,
                warm_start=True,
                verbose=solver_verbose,
                **solve_options,
            )
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
