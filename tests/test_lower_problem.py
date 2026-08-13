"""Tests for lower-level variable-to-parameter modeling."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy.errors import ParameterMappingError
from blvpy.lower_problem import LowerProblem


def test_replaces_linked_scalar_and_preserves_lower_variable_identity() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    objective = cp.Minimize(cp.square(y - x))
    constraint = y >= x

    lower = LowerProblem(objective, [constraint], parameters=[x])

    assert lower.objective is objective
    assert lower.constraints == (constraint,)
    assert lower.parameters == (x,)
    assert lower.linked_variables == (x,)
    assert lower._cvxpy_problem.objective is not objective
    assert lower._cvxpy_problem.variables() == [y]
    assert lower._cvxpy_problem.variables()[0] is y

    internal_parameter = lower._internal_parameters[0]
    assert internal_parameter.name() == "x_lower"
    assert lower._cvxpy_problem.parameters() == [internal_parameter]
    assert lower._parameter_map[internal_parameter] is x
    assert tuple(lower._parameter_map) == (internal_parameter,)


def test_replaces_vector_and_matrix_variables_with_matching_parameters() -> None:
    x = cp.Variable(2, nonneg=True, name="x")
    matrix = cp.Variable((2, 2), symmetric=True, name="matrix")
    y = cp.Variable(2, name="y")
    objective = cp.Minimize(cp.sum_squares(y - x) + cp.sum_squares(matrix @ y))

    lower = LowerProblem(objective, parameters=[x, matrix])

    vector_parameter, matrix_parameter = lower._internal_parameters
    assert vector_parameter.shape == x.shape
    assert vector_parameter.attributes["nonneg"]
    assert matrix_parameter.shape == matrix.shape
    assert matrix_parameter.attributes["symmetric"]
    assert set(lower._cvxpy_problem.variables()) == {y}
    assert lower._parameter_map[vector_parameter] is x
    assert lower._parameter_map[matrix_parameter] is matrix


def test_copies_bounds_without_aliasing_variable_metadata() -> None:
    x = cp.Variable(2, bounds=[-1.0, 2.0], name="x")
    y = cp.Variable(2, name="y")

    lower = LowerProblem(cp.Minimize(cp.sum_squares(y - x)), parameters=[x])
    parameter = lower._internal_parameters[0]

    np.testing.assert_allclose(parameter.bounds[0], [-1.0, -1.0])
    np.testing.assert_allclose(parameter.bounds[1], [2.0, 2.0])
    assert parameter.bounds is not x.bounds
    assert parameter.bounds[0] is not x.bounds[0]
    assert parameter.bounds[1] is not x.bounds[1]


def test_keeps_fixed_cvxpy_parameters_and_multiple_lower_variables() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    z = cp.Variable(name="z")
    fixed = cp.Parameter(value=2.0, name="fixed")
    lower = LowerProblem(
        cp.Minimize(cp.square(y - x) + cp.square(z - fixed)),
        [y + z >= x],
        parameters=[x],
    )

    assert lower._cvxpy_problem.variables() == [y, z]
    assert all(actual is expected for actual, expected in zip(lower._cvxpy_problem.variables(), (y, z), strict=True))
    internal_ids = {id(parameter) for parameter in lower._cvxpy_problem.parameters()}
    assert id(fixed) in internal_ids
    assert id(lower._internal_parameters[0]) in internal_ids


def test_internal_problem_data_tracks_linked_values_without_mutating_source() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    objective = cp.Minimize(cp.square(y - x))
    constraint = y >= x
    lower = LowerProblem(objective, [constraint], parameters=[x])
    parameter = lower._internal_parameters[0]

    parameter.value = 1.5
    assert lower._cvxpy_problem.solve(solver=cp.CLARABEL) == pytest.approx(0.0, abs=1e-8)
    assert y.value == pytest.approx(1.5, abs=1e-4)
    assert x.value is None
    assert any(variable is x for variable in objective.variables())
    assert any(variable is x for variable in constraint.variables())


def test_accepts_linked_variable_used_only_in_constraints() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    lower = LowerProblem(cp.Minimize(cp.square(y)), [y >= x], parameters=[x])

    assert lower.parameters == (x,)
    assert lower._cvxpy_problem.variables() == [y]


def test_rejects_duplicate_linked_variable() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    with pytest.raises(ParameterMappingError, match="x.*listed more than once"):
        LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x, x])


@pytest.mark.parametrize("invalid", [1.0, cp.Parameter(name="p"), cp.Constant(1.0)])
def test_rejects_nonvariable_link(invalid: object) -> None:
    y = cp.Variable(name="y")

    with pytest.raises(TypeError, match="must be a CVXPY Variable"):
        LowerProblem(cp.Minimize(cp.square(y)), parameters=[invalid])  # type: ignore[list-item]


def test_rejects_linked_variable_absent_from_lower_expressions() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    with pytest.raises(ParameterMappingError, match="x.*does not occur"):
        LowerProblem(cp.Minimize(cp.square(y)), parameters=[x])


def test_rejects_invalid_objective_and_constraints() -> None:
    y = cp.Variable(name="y")

    with pytest.raises(TypeError, match="objective must be a CVXPY Objective"):
        LowerProblem(cp.square(y))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Every lower constraint"):
        LowerProblem(cp.Minimize(cp.square(y)), [True])  # type: ignore[list-item]


def test_none_constraints_and_parameters_match_empty_sequences() -> None:
    y = cp.Variable(name="y")

    lower = LowerProblem(cp.Minimize(cp.square(y)), constraints=None, parameters=None)

    assert lower.constraints == ()
    assert lower.parameters == ()
    assert lower._cvxpy_problem.variables() == [y]
