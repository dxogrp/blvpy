"""Lower-level problem modeling with implicit upper-variable parameters."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import cvxpy as cp

from .errors import ParameterMappingError


class LowerProblem:
    """A convex lower problem parameterized by selected upper variables.

    ``LowerProblem`` has the same basic construction pattern as
    :class:`cvxpy.Problem`, with the additional ``parameters`` argument. Each
    variable listed there is an upper variable: BLVPY replaces it internally
    by a CVXPY parameter while retaining every unlisted variable as a lower
    decision variable. The supplied expression trees are not mutated.

    Parameters
    ----------
    objective : cvxpy.Objective
        Objective of the lower problem. A supported bilevel model must use
        :class:`cvxpy.Minimize` or :class:`cvxpy.Maximize` and satisfy DCP and
        DPP when validated. Maximization objectives are normalized internally
        to equivalent minimization objectives during canonicalization.
    constraints : sequence of cvxpy.Constraint or None, optional
        Lower constraints. ``None`` and the default empty sequence both mean
        that the lower problem has no explicit constraints.
    parameters : sequence of cvxpy.Variable or None, optional
        Upper variables that are held fixed when solving the lower problem.
        Every listed variable must occur in ``objective`` or ``constraints``.
        ``None`` and the default empty sequence mean that no upper variables
        occur in the lower model.

    Raises
    ------
    TypeError
        If the objective, constraints, or parameters have invalid types.
    ParameterMappingError
        If a parameter variable is duplicated or does not occur in the lower
        expressions.

    Notes
    -----
    Domain attributes and native bounds on a listed variable are copied to its
    generated parameter. Lower decision variables remain the same CVXPY
    objects used by the upper expressions, which implements optimistic
    selection among multiple lower optima.
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
        parameter_variables = _validate_parameters(parameters)
        _validate_parameter_usage(objective, constraint_tuple, parameter_variables)

        replacements: dict[int, cp.Parameter] = {}
        parameter_links: dict[cp.Parameter, cp.Variable] = {}
        internal_parameters: list[cp.Parameter] = []
        for variable in parameter_variables:
            parameter = _parameter_for(variable)
            replacements[id(variable)] = parameter
            parameter_links[parameter] = variable
            internal_parameters.append(parameter)

        internal_objective = objective.tree_copy(replacements)
        internal_constraints = tuple(constraint.tree_copy(replacements) for constraint in constraint_tuple)

        self._objective = objective
        self._constraints = constraint_tuple
        self._parameters = parameter_variables
        self._cvxpy_problem = cp.Problem(internal_objective, list(internal_constraints))
        self._parameter_links = MappingProxyType(parameter_links)
        self._internal_parameters = tuple(internal_parameters)

    @property
    def objective(self) -> cp.Objective:
        """cvxpy.Objective: The original lower objective."""

        return self._objective

    @property
    def constraints(self) -> tuple[cp.Constraint, ...]:
        """tuple of cvxpy.Constraint: Original lower constraints in construction order."""

        return self._constraints

    @property
    def parameters(self) -> tuple[cp.Variable, ...]:
        """tuple of cvxpy.Variable: Upper variables treated as lower parameters."""

        return self._parameters


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
        parameter_variables = tuple(parameters)
    except TypeError as error:
        raise TypeError("parameters must be a sequence of CVXPY variables.") from error

    seen: set[int] = set()
    for variable in parameter_variables:
        if not isinstance(variable, cp.Variable):
            raise TypeError("Every LowerProblem parameter must be a CVXPY Variable.")
        identity = id(variable)
        if identity in seen:
            raise ParameterMappingError(f"LowerProblem parameter {variable.name()!r} is listed more than once.")
        seen.add(identity)
    return parameter_variables


def _validate_parameter_usage(
    objective: cp.Objective,
    constraints: tuple[cp.Constraint, ...],
    parameter_variables: tuple[cp.Variable, ...],
) -> None:
    used_ids = {id(variable) for variable in objective.variables()}
    for constraint in constraints:
        used_ids.update(id(variable) for variable in constraint.variables())
    for variable in parameter_variables:
        if id(variable) not in used_ids:
            raise ParameterMappingError(
                f"LowerProblem parameter {variable.name()!r} does not occur in the lower problem."
            )


def _parameter_for(variable: cp.Variable) -> cp.Parameter:
    attributes: dict[str, Any] = {
        name: deepcopy(value) for name, value in variable.attributes.items() if value is not False and value is not None
    }
    return cp.Parameter(
        variable.shape,
        name=f"{variable.name()}_lower",
        **attributes,
    )
