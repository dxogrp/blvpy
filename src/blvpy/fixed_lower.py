"""Reusable fixed-upper canonical lower solves."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .backends import solve_conic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .problem import BilevelProblem


class FixedLowerSolveError(RuntimeError):
    """Internal failure to obtain a complete fixed-upper conic certificate."""


@dataclass(frozen=True, slots=True)
class FixedLowerSolution:
    """Numerical certificate and recovered values from one canonical solve."""

    status: str
    primal: NDArray[np.float64]
    slack: NDArray[np.float64]
    dual: NDArray[np.float64]
    source_values: Mapping[int, NDArray[np.float64]]

    def __post_init__(self) -> None:
        for name in ("primal", "slack", "dual"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        source_values: dict[int, NDArray[np.float64]] = {}
        for variable_id, value in self.source_values.items():
            snapshot = np.array(value, dtype=float, copy=True)
            snapshot.setflags(write=False)
            source_values[variable_id] = snapshot
        object.__setattr__(self, "source_values", MappingProxyType(source_values))


def solve_fixed_lower(
    model: BilevelProblem,
    solver: str,
    options: Mapping[str, object],
    solver_verbose: bool,
) -> FixedLowerSolution:
    """Solve the evaluated canonical lower problem without mutating lifted state."""

    canonical = model.canonicalize()
    data = canonical.apply_numeric()
    primal = cp.Variable(canonical.canonical_size, name="blvpy_fixed_lower_primal")
    slack = cp.Variable(canonical.constraint_size, name="blvpy_fixed_lower_slack")
    equality = data.A @ primal + slack == data.b
    problem = cp.Problem(
        cp.Minimize(data.c @ primal),
        [equality, *canonical.cone_layout.primal_constraints(slack)],
    )
    solve_conic(problem, solver, options, solver_verbose)
    if problem.status not in cp.settings.SOLUTION_PRESENT:
        raise FixedLowerSolveError(f"The fixed-upper lower cone problem returned status {problem.status!r}.")
    if primal.value is None or slack.value is None or equality.dual_value is None:
        raise FixedLowerSolveError("The fixed-upper lower conic solver omitted a primal or dual certificate.")

    primal_value = np.asarray(primal.value, dtype=float)
    slack_value = np.asarray(slack.value, dtype=float)
    dual_value = np.asarray(equality.dual_value, dtype=float)
    if not all(np.isfinite(value).all() for value in (primal_value, slack_value, dual_value)):
        raise FixedLowerSolveError("The fixed-upper lower conic solver returned a nonfinite certificate.")
    return FixedLowerSolution(
        status=str(problem.status),
        primal=primal_value,
        slack=slack_value,
        dual=dual_value,
        source_values=canonical.recover_numeric(primal_value),
    )


__all__: list[str] = []
