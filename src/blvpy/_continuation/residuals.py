"""Independent residual calculations for lifted continuation points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import InitializationError
from ..result import Residuals

if TYPE_CHECKING:
    from ..problem import BilevelProblem


def compute_residuals(model: BilevelProblem, epsilon: float | None = None) -> Residuals:
    """Independently compute all reported lifted residuals."""

    lifted = model._lifted_problem
    if epsilon is None:
        epsilon = float(lifted.epsilon.value)
    values = {parameter: variable.value for parameter, variable in model._parameter_links.items()}
    if any(value is None for value in values.values()):
        raise InitializationError("Linked upper variables do not all have numeric values.")
    data = model.canonicalize().apply_numeric(values)
    primal = _required_vector(lifted.primal.value, "canonical primal")
    slack = _required_vector(lifted.slack.value, "canonical slack")
    dual = _required_vector(lifted.dual.value, "canonical dual")
    primal_residual = data.A @ primal + slack - data.b
    dual_residual = data.A.T @ dual + data.c

    recovered = model.canonicalize().recover_numeric(primal)
    recovery = 0.0
    lower_by_id = {variable.id: variable for variable in model._cvxpy_lower_problem.variables()}
    for variable_id, expected in recovered.items():
        actual = lower_by_id[variable_id].value
        if actual is None:
            recovery = float("inf")
            break
        recovery = max(recovery, _norm(np.asarray(actual) - expected))
    upper = _constraint_violation(lifted.upper_constraints)
    complementarity = model.canonicalize().cone_layout.complementarity(slack, dual)
    return Residuals(
        primal_equality=_norm(primal_residual),
        dual_equality=_norm(dual_residual),
        recovery=recovery,
        upper_constraints=upper,
        primal_cone=model.canonicalize().cone_layout.primal_distance(slack),
        dual_cone=model.canonicalize().cone_layout.dual_distance(dual),
        complementarity=complementarity,
        gap_violation=max(complementarity - float(epsilon), 0.0),
    )


def _constraint_violation(constraints) -> float:
    violation = 0.0
    for constraint in constraints:
        try:
            value = np.asarray(constraint.violation(), dtype=float)
        except Exception:
            return float("inf")
        violation = max(violation, _norm(value))
    return violation


def _finite_constraint_violation(constraints) -> float:
    """Return the largest finite violation when another constraint is outside its domain."""

    violation = 0.0
    for constraint in constraints:
        try:
            value = _norm(np.asarray(constraint.violation(), dtype=float))
        except Exception:
            continue
        if np.isfinite(value):
            violation = max(violation, value)
    return violation


def _required_vector(value: Any, name: str) -> NDArray[np.float64]:
    if value is None:
        raise InitializationError(f"The {name} has no numeric value.")
    return np.asarray(value, dtype=float).reshape(-1, order="F")


def _norm(value: ArrayLike) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        return float("inf")
    return float(np.linalg.norm(array))


def _infinite_residuals() -> Residuals:
    return Residuals(
        primal_equality=float("inf"),
        dual_equality=float("inf"),
        recovery=float("inf"),
        upper_constraints=float("inf"),
        primal_cone=float("inf"),
        dual_cone=float("inf"),
        complementarity=float("inf"),
        gap_violation=float("inf"),
    )
