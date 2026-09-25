"""Upper-variable sampling helpers for continuation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import InitializationError
from .state import _numeric_value

if TYPE_CHECKING:
    from ..problem import BilevelProblem


def _generate_upper_initializations(
    model: BilevelProblem,
    best_of: int | None,
    rng: np.random.Generator,
) -> tuple[dict[cp.Variable, NDArray[np.float64]], ...]:
    """Generate deterministic or CVXPY-style randomized upper points."""

    if best_of is None:
        sample: dict[cp.Variable, NDArray[np.float64]] = {}
        for variable in model.upper_variables:
            if variable.value is not None:
                value = _numeric_value(variable.value, variable.shape)
            else:
                lower, upper = _variable_bounds(variable)
                both = np.isfinite(lower) & np.isfinite(upper)
                lower_only = np.isfinite(lower) & ~np.isfinite(upper)
                upper_only = ~np.isfinite(lower) & np.isfinite(upper)
                value = np.zeros(variable.shape, dtype=float)
                value[both] = (lower[both] + upper[both]) / 2.0
                value[lower_only] = lower[lower_only] + 1.0
                value[upper_only] = upper[upper_only] - 1.0
                value = _project_variable_value(variable, value)
            sample[variable] = value
        return (sample,)

    specifications: list[
        tuple[
            cp.Variable,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
        ]
    ] = []
    missing: list[str] = []
    for variable in model.upper_variables:
        sample_bounds = getattr(variable, "sample_bounds", None)
        if sample_bounds is not None:
            lower, upper = _validated_sample_bounds(variable, sample_bounds)
            specifications.append((variable, None, lower, upper))
        elif variable.value is not None:
            specifications.append(
                (
                    variable,
                    _numeric_value(variable.value, variable.shape),
                    None,
                    None,
                )
            )
        else:
            lower, upper = _variable_bounds(variable)
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
                specifications.append((variable, None, lower, upper))
            else:
                missing.append(variable.name())
    if missing:
        names = ", ".join(missing)
        raise InitializationError(
            "Random best-of initialization requires .value, finite "
            ".sample_bounds, or finite native bounds for variables: "
            f"{names}."
        )

    samples: list[dict[cp.Variable, NDArray[np.float64]]] = []
    for _ in range(best_of):
        sample = {}
        for variable, fixed, lower, upper in specifications:
            if fixed is not None:
                value = fixed.copy()
            else:
                assert lower is not None and upper is not None
                value = np.asarray(
                    rng.uniform(lower, upper),
                    dtype=float,
                ).reshape(variable.shape, order="F")
                value = _project_variable_value(variable, value)
            sample[variable] = value
        samples.append(sample)
    return tuple(samples)


def _validated_sample_bounds(
    variable: cp.Variable,
    sample_bounds: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        lower_value, upper_value = sample_bounds
    except (TypeError, ValueError) as error:
        raise ValueError(f"sample_bounds for variable {variable.name()!r} must be a (lower, upper) pair.") from error
    if np.iscomplexobj(lower_value) or np.iscomplexobj(upper_value):
        raise ValueError(f"sample_bounds for variable {variable.name()!r} must be real.")
    try:
        lower = np.broadcast_to(
            np.asarray(lower_value, dtype=float),
            variable.shape,
        ).copy()
        upper = np.broadcast_to(
            np.asarray(upper_value, dtype=float),
            variable.shape,
        ).copy()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"sample_bounds for variable {variable.name()!r} cannot be broadcast to shape {variable.shape}."
        ) from error
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(f"sample_bounds for variable {variable.name()!r} must be finite.")
    if np.any(lower > upper):
        raise ValueError(
            f"sample_bounds for variable {variable.name()!r} have lower entries greater than upper entries."
        )
    return lower, upper


def _variable_bounds(
    variable: cp.Variable,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower, upper = variable.get_bounds()
    lower_array = np.broadcast_to(np.asarray(lower, dtype=float), variable.shape).copy()
    upper_array = np.broadcast_to(np.asarray(upper, dtype=float), variable.shape).copy()
    if np.any(lower_array > upper_array):
        raise InitializationError(f"Variable {variable.name()!r} has inconsistent bounds.")
    return lower_array, upper_array


def _project_variable_value(
    variable: cp.Variable,
    value: ArrayLike,
) -> NDArray[np.float64]:
    try:
        projected = variable.project(value)
    except (TypeError, ValueError) as error:
        raise InitializationError(
            f"Could not project an automatic value for upper variable {variable.name()!r} onto its declared attributes."
        ) from error
    if hasattr(projected, "toarray"):
        projected = projected.toarray()
    return _numeric_value(projected, variable.shape)
