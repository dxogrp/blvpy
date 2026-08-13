"""Lower-level problem modeling with implicit upper-variable parameters."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import cvxpy as cp

from .errors import ParameterMappingError


class LowerProblem:
    """A CVXPY lower problem parameterized by selected upper variables.

    The public objective and constraints retain the expressions supplied by the
    user. Internally, every variable listed in ``parameters`` is replaced by a
    CVXPY parameter with the same shape and leaf-domain attributes. Variables
    not listed in ``parameters`` remain the original CVXPY objects.
    """

    def __init__(
        self,
        objective: cp.Objective,
        constraints: Sequence[cp.Constraint] | None = (),
        parameters: Sequence[cp.Variable] | None = (),
    ) -> None:
        if not isinstance(objective, cp.Objective):
            raise TypeError("objective must be a CVXPY Objective.")
        constraint_tuple = _validate_constraints(constraints)
        linked_variables = _validate_parameters(parameters)
        _validate_parameter_usage(objective, constraint_tuple, linked_variables)

        replacements: dict[int, cp.Parameter] = {}
        parameter_map: dict[cp.Parameter, cp.Variable] = {}
        internal_parameters: list[cp.Parameter] = []
        for variable in linked_variables:
            parameter = _parameter_for(variable)
            replacements[id(variable)] = parameter
            parameter_map[parameter] = variable
            internal_parameters.append(parameter)

        internal_objective = objective.tree_copy(replacements)
        internal_constraints = tuple(constraint.tree_copy(replacements) for constraint in constraint_tuple)

        self._objective = objective
        self._constraints = constraint_tuple
        self._linked_variables = linked_variables
        self._cvxpy_problem = cp.Problem(internal_objective, list(internal_constraints))
        self._parameter_map = MappingProxyType(parameter_map)
        self._internal_parameters = tuple(internal_parameters)

    @property
    def objective(self) -> cp.Objective:
        """The original lower-level objective."""

        return self._objective

    @property
    def constraints(self) -> tuple[cp.Constraint, ...]:
        """The original lower-level constraints in construction order."""

        return self._constraints

    @property
    def parameters(self) -> tuple[cp.Variable, ...]:
        """Upper variables treated as parameters by the lower problem."""

        return self._linked_variables

    @property
    def linked_variables(self) -> tuple[cp.Variable, ...]:
        """Alias for the upper variables declared through ``parameters``."""

        return self._linked_variables


def _validate_constraints(
    constraints: Sequence[cp.Constraint] | None,
) -> tuple[cp.Constraint, ...]:
    if constraints is None:
        return ()
    try:
        constraint_tuple = tuple(constraints)
    except TypeError as error:
        raise TypeError("constraints must be a sequence of CVXPY constraints.") from error
    if not all(isinstance(constraint, cp.Constraint) for constraint in constraint_tuple):
        raise TypeError("Every lower constraint must be a CVXPY Constraint.")
    return constraint_tuple


def _validate_parameters(
    parameters: Sequence[cp.Variable] | None,
) -> tuple[cp.Variable, ...]:
    if parameters is None:
        return ()
    try:
        linked_variables = tuple(parameters)
    except TypeError as error:
        raise TypeError("parameters must be a sequence of CVXPY variables.") from error

    seen: set[int] = set()
    for variable in linked_variables:
        if not isinstance(variable, cp.Variable):
            raise TypeError("Every LowerProblem parameter must be a CVXPY Variable.")
        identity = id(variable)
        if identity in seen:
            raise ParameterMappingError(f"Linked variable {variable.name()!r} is listed more than once.")
        seen.add(identity)
    return linked_variables


def _validate_parameter_usage(
    objective: cp.Objective,
    constraints: tuple[cp.Constraint, ...],
    linked_variables: tuple[cp.Variable, ...],
) -> None:
    used_ids = {id(variable) for variable in objective.variables()}
    for constraint in constraints:
        used_ids.update(id(variable) for variable in constraint.variables())
    for variable in linked_variables:
        if id(variable) not in used_ids:
            raise ParameterMappingError(f"Linked variable {variable.name()!r} does not occur in the lower problem.")


def _parameter_for(variable: cp.Variable) -> cp.Parameter:
    attributes: dict[str, Any] = {
        name: deepcopy(value) for name, value in variable.attributes.items() if value is not False and value is not None
    }
    return cp.Parameter(
        variable.shape,
        name=f"{variable.name()}_lower",
        **attributes,
    )
