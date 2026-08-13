"""Solver-independent tests for the public bilevel modeling interface."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy.errors import (
    ParameterMappingError,
    UnsupportedModelError,
    ValidationError,
)
from blvpy.problem import BilevelProblem


def _quadratic_bilevel(*, bounds: tuple[float, float] = (-2.0, 2.0)):
    x = cp.Variable(name="x", bounds=list(bounds))
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="lower_x")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    problem = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
        {parameter: x},
        [x + y <= 3.0],
    )
    return problem, x, y, parameter


def test_valid_problem_caches_canonical_and_lifted_models() -> None:
    problem, x, y, parameter = _quadratic_bilevel()

    assert problem.is_dbp()
    assert problem.validate() is None
    assert problem.canonicalize() is problem.canonicalize()
    assert problem.lifted_problem is problem.lifted_problem
    assert problem.upper_variables == (x,)
    assert problem.source_variables == (x, y)
    assert parameter in problem.parameter_map


def test_lifted_problem_contains_full_optimistic_reformulation() -> None:
    problem, _, y, _ = _quadratic_bilevel()
    canonical = problem.canonicalize()
    lifted = problem.lifted_problem

    assert lifted.problem.objective is problem.outer_objective
    assert lifted.problem.is_dnlp()
    assert lifted.primal.shape == (canonical.canonical_size,)
    assert lifted.slack.shape == (canonical.constraint_size,)
    assert lifted.dual.shape == (canonical.constraint_size,)
    assert lifted.primal_equality in lifted.problem.constraints
    assert lifted.dual_equality in lifted.problem.constraints
    assert lifted.gap_constraint in lifted.problem.constraints
    assert y.id in lifted.recovery_expressions
    assert len(lifted.recovery_constraints) == 1

    lifted.epsilon.value = 2.5e-4
    assert lifted.epsilon.value == pytest.approx(2.5e-4)


def test_lifted_affine_data_tracks_mapped_parameter() -> None:
    x = cp.Variable(name="x", bounds=[0.5, 2.0])
    y = cp.Variable(2, name="y")
    t = cp.Variable(name="t")
    parameter = cp.Parameter(nonneg=True, name="lower_x")
    lower = cp.Problem(
        cp.Minimize(t + parameter * y[0]),
        [cp.norm(y, 2) <= t, parameter * y[0] + y[1] >= 1.0],
    )
    problem = BilevelProblem(cp.Minimize(cp.square(x - 1.0) + t), lower, {parameter: x})
    canonical = problem.canonicalize()
    expressions = problem.lifted_problem.canonical_expressions

    for value in (0.5, 1.25, 2.0):
        x.value = value
        expected = canonical.apply_numeric({parameter: value})
        np.testing.assert_allclose(expressions.A.value, expected.A.toarray())
        np.testing.assert_allclose(expressions.b.value, expected.b)
        np.testing.assert_allclose(expressions.c.value, expected.c)
        assert float(expressions.d.value) == pytest.approx(expected.d)


def test_mapped_parameter_domain_is_carried_into_lifted_problem() -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    parameter = cp.Parameter(nonneg=True, name="nonnegative_parameter")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    problem = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower, {parameter: x})

    lifted = problem.lifted_problem
    assert len(lifted.upper_constraints) == 1
    x.value = -1.0
    assert float(np.max(lifted.upper_constraints[0].violation())) == pytest.approx(1.0)
    x.value = 1.0
    assert float(np.max(lifted.upper_constraints[0].violation())) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: (
                cp.Maximize(cp.Variable(name="upper_x")),
                cp.Problem(cp.Minimize(cp.Variable(name="lower_y"))),
                {},
            ),
            "upper problem must be a minimization",
        ),
        (
            lambda: (
                cp.Minimize(cp.Variable(name="upper_x")),
                cp.Problem(cp.Maximize(cp.Variable(name="lower_y"))),
                {},
            ),
            "lower problem must be a minimization",
        ),
    ],
)
def test_rejects_maximization_levels(factory, message: str) -> None:
    objective, lower, mapping = factory()
    problem = BilevelProblem(objective, lower, mapping)

    assert not problem.is_dbp()
    with pytest.raises(ValidationError, match=message):
        problem.validate()


def test_rejects_missing_fixed_parameter_value() -> None:
    x = cp.Variable(name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(name="y")
    linked = cp.Parameter(name="linked")
    fixed = cp.Parameter(name="fixed")
    lower = cp.Problem(cp.Minimize(cp.square(y - linked) + fixed * y))
    problem = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower, {linked: x})

    assert not problem.is_dbp()
    with pytest.raises(ParameterMappingError, match="fixed value"):
        problem.validate()


def test_rejects_parameter_map_shape_mismatch() -> None:
    x = cp.Variable(2, name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="scalar_parameter")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    problem = BilevelProblem(cp.Minimize(cp.sum_squares(x) + cp.square(y)), lower, {parameter: x})

    assert not problem.is_dbp()
    with pytest.raises(ValidationError, match="shape"):
        problem.validate()


def test_rejects_nonvariable_parameter_link() -> None:
    x = cp.Variable(name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="parameter")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    problem = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)), lower, {parameter: x + 1.0}  # type: ignore[dict-item]
    )

    assert not problem.is_dbp()
    with pytest.raises(ValidationError, match="must be a CVXPY Variable"):
        problem.validate()


def test_rejects_upper_atom_that_is_not_dnlp() -> None:
    x = cp.Variable(name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(name="y")
    parameter = cp.Parameter(name="parameter")
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))
    problem = BilevelProblem(cp.Minimize(cp.ceil(x) + cp.square(y)), lower, {parameter: x})

    assert not problem.is_dbp()
    with pytest.raises(UnsupportedModelError, match="not DNLP compliant"):
        problem.validate()


def test_constructor_rejects_non_cvxpy_outer_constraints() -> None:
    _, _, y, parameter = _quadratic_bilevel()
    lower = cp.Problem(cp.Minimize(cp.square(y - parameter)))

    with pytest.raises(TypeError, match="CVXPY Constraint"):
        BilevelProblem(cp.Minimize(cp.square(y)), lower, {}, [True])  # type: ignore[list-item]


def test_solve_defaults_match_documented_continuation_settings() -> None:
    import inspect

    parameters = inspect.signature(BilevelProblem.solve).parameters
    assert parameters["epsilon_initial"].default == pytest.approx(1e-1)
    assert parameters["epsilon_target"].default == pytest.approx(1e-6)
    assert parameters["contraction"].default == pytest.approx(0.1)
    assert parameters["starts"].default == 10
    assert parameters["feasibility_tolerance"].default == pytest.approx(1e-7)
    assert parameters["solver"].default == cp.IPOPT
    assert parameters["conic_solver"].default == cp.CLARABEL
