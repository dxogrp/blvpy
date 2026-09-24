"""Numeric state management for continuation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import InitializationError

if TYPE_CHECKING:
    from ..problem import BilevelProblem


def _sync_linked_parameters(model: BilevelProblem) -> None:
    for parameter, variable in model._parameter_links.items():
        if variable.value is None:
            raise InitializationError(f"Linked upper variable {variable.name()!r} has no value.")
        parameter.value = np.asarray(variable.value, dtype=float)


def _snapshot_state(problem: cp.Problem) -> dict[int, NDArray[np.float64]]:
    state: dict[int, NDArray[np.float64]] = {}
    for variable in problem.variables():
        if variable.value is None:
            raise InitializationError(f"NLP solve did not return a value for variable {variable.name()!r}.")
        state[variable.id] = np.array(variable.value, dtype=float, copy=True)
    return state


def _restore_state(problem: cp.Problem, state: Mapping[int, ArrayLike]) -> None:
    for variable in problem.variables():
        if variable.id in state:
            variable.save_value(np.array(state[variable.id], dtype=float, copy=True))


def _assign_values(values: Mapping[cp.Variable, ArrayLike]) -> None:
    for variable, value in values.items():
        variable.project_and_assign(value)


def _numeric_value(value: Any, shape: tuple[int, ...]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        array = np.reshape(array, shape, order="F")
    if not np.all(np.isfinite(array)):
        raise InitializationError("Initial upper-variable values must be finite.")
    return array.copy()
