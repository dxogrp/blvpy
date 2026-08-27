"""Solver-independent tests for the public bilevel modeling interface."""

from __future__ import annotations

import inspect
import warnings

import cvxpy as cp
import numpy as np
import pytest

from blvpy import BilevelProblem, LowerProblem
from blvpy.errors import (
    ParameterMappingError,
    UnsupportedModelError,
    ValidationError,
)


def _quadratic_bilevel(*, bounds: tuple[float, float] = (-2.0, 2.0)):
    x = cp.Variable(name="x", bounds=list(bounds))
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
        upper_constraints=[x + y <= 3.0],
    )
    parameter = next(iter(problem._parameter_links))
    return problem, x, y, parameter


def test_valid_problem_caches_canonical_and_lifted_models() -> None:
    problem, x, y, parameter = _quadratic_bilevel()

    assert problem.is_dblp()
    assert problem.validate() is None
    assert problem.canonicalize() is problem.canonicalize()
    assert problem._lifted_problem is problem._lifted_problem
    assert problem.upper_variables == (x,)
    assert problem.source_variables == (x, y)
    assert parameter in problem._parameter_links


def test_lifted_problem_contains_full_optimistic_reformulation() -> None:
    problem, _, y, _ = _quadratic_bilevel()
    canonical = problem.canonicalize()
    lifted = problem._lifted_problem

    assert lifted.problem.objective is problem.upper_objective
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
    lower = LowerProblem(
        cp.Minimize(t + x * y[0]),
        [cp.norm(y, 2) <= t, x * y[0] + y[1] >= 1.0],
        parameters=[x],
    )
    problem = BilevelProblem(cp.Minimize(cp.square(x - 1.0) + t), lower)
    parameter = next(iter(problem._parameter_links))
    canonical = problem.canonicalize()
    expressions = problem._lifted_problem.canonical_expressions

    for value in (0.5, 1.25, 2.0):
        x.value = value
        expected = canonical.apply_numeric({parameter: value})
        np.testing.assert_allclose(expressions.A.value, expected.A.toarray())
        np.testing.assert_allclose(expressions.b.value, expected.b)
        np.testing.assert_allclose(expressions.c.value, expected.c)
        assert float(expressions.d.value) == pytest.approx(expected.d)


def test_demand_response_lifted_data_is_compact_and_tracks_price() -> None:
    hours = np.arange(24, dtype=float)
    preferred_load = (
        1.2 + 0.35 * np.sin(2.0 * np.pi * (hours - 7.0) / 24.0) + 0.55 * np.exp(-0.5 * ((hours - 19.0) / 2.2) ** 2)
    )
    daily_energy = float(np.sum(preferred_load))

    price = cp.Variable(24, nonneg=True, bounds=[0.0, 0.8], name="hourly_price")
    peak_load = cp.Variable(nonneg=True, bounds=[0.0, 4.0], name="peak_load")
    load = cp.Variable(24, name="hourly_load")
    lower = LowerProblem(
        cp.Minimize(price @ load + 0.8 * cp.sum_squares(load - preferred_load)),
        [
            cp.sum(load) == daily_energy,
            load >= 0.2,
            load <= 2.5,
        ],
        parameters=[price],
    )
    problem = BilevelProblem(
        cp.Minimize(peak_load + 0.025 * cp.sum_squares(price) + 0.12 * cp.sum_squares(price[1:] - price[:-1])),
        lower,
        upper_constraints=[load <= peak_load],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem.validate()

    assert not any("too many subexpressions" in str(item.message).lower() for item in caught)

    canonical = problem.canonicalize()
    lifted = problem._lifted_problem
    expressions = lifted.canonical_expressions
    generated_parameter = next(iter(problem._parameter_links))

    assert lifted.problem.is_dnlp()
    assert expressions.A.shape == (canonical.constraint_size, canonical.canonical_size)
    assert expressions.b.shape == (canonical.constraint_size,)
    assert expressions.c.shape == (canonical.canonical_size,)
    assert expressions.d.shape == ()
    assert lifted.primal_equality.shape == (canonical.constraint_size,)
    assert lifted.dual_equality.shape == (canonical.canonical_size,)

    for value in (np.zeros(24), np.linspace(0.05, 0.75, 24)):
        price.value = value
        expected = canonical.apply_numeric({generated_parameter: value})
        np.testing.assert_allclose(expressions.A.value, expected.A.toarray())
        np.testing.assert_allclose(expressions.b.value, expected.b)
        np.testing.assert_allclose(expressions.c.value, expected.c)
        assert float(expressions.d.value) == pytest.approx(expected.d)


def test_mapped_parameter_domain_is_carried_into_lifted_problem() -> None:
    x = cp.Variable(name="x", nonneg=True)
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower)

    lifted = problem._lifted_problem
    assert len(lifted.upper_constraints) == 1
    x.save_value(-1.0)
    assert float(np.max(lifted.upper_constraints[0].violation())) == pytest.approx(1.0)
    x.value = 1.0
    assert float(np.max(lifted.upper_constraints[0].violation())) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("upper_sense", "lower_sense"),
    [
        (cp.Minimize, cp.Minimize),
        (cp.Minimize, cp.Maximize),
        (cp.Maximize, cp.Minimize),
        (cp.Maximize, cp.Maximize),
    ],
    ids=("min-min", "min-max", "max-min", "max-max"),
)
def test_accepts_all_upper_and_lower_objective_sense_combinations(
    upper_sense,
    lower_sense,
) -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower_expression = cp.square(y - x)
    lower_objective = lower_sense(lower_expression) if lower_sense is cp.Minimize else lower_sense(-lower_expression)
    upper_expression = cp.square(x) + cp.square(y)
    upper_objective = upper_sense(upper_expression) if upper_sense is cp.Minimize else upper_sense(-upper_expression)
    lower = LowerProblem(lower_objective, parameters=[x])
    problem = BilevelProblem(upper_objective, lower)

    assert problem.is_dblp()
    assert problem.validate() is None
    assert problem.upper_objective is upper_objective
    assert problem.lower_problem.objective is lower_objective
    assert problem._lifted_problem.problem.objective is upper_objective
    assert isinstance(problem._cvxpy_lower_problem.objective, lower_sense)


def test_rejects_convex_lower_maximization_as_non_dcp() -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Maximize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower)

    assert not problem.is_dblp()
    with pytest.raises(ValidationError, match="DCP"):
        problem.validate()


def test_rejects_missing_fixed_parameter_value() -> None:
    x = cp.Variable(name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(name="y")
    fixed = cp.Parameter(name="fixed")
    lower = LowerProblem(cp.Minimize(cp.square(y - x) + fixed * y), parameters=[x])
    problem = BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), lower)

    assert not problem.is_dblp()
    with pytest.raises(ParameterMappingError, match="fixed value"):
        problem.validate()


def test_rejects_upper_atom_that_is_not_dnlp() -> None:
    x = cp.Variable(name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(cp.Minimize(cp.ceil(x) + cp.square(y)), lower)

    assert not problem.is_dblp()
    with pytest.raises(UnsupportedModelError, match="not DNLP compliant"):
        problem.validate()


def test_constructor_rejects_non_cvxpy_upper_constraints() -> None:
    _, _, y, _ = _quadratic_bilevel()
    lower = LowerProblem(cp.Minimize(cp.square(y)))

    with pytest.raises(TypeError, match="CVXPY Constraint"):
        BilevelProblem(cp.Minimize(cp.square(y)), lower, upper_constraints=[True])  # type: ignore[list-item]


def test_constructor_accepts_lower_problem_and_rejects_raw_cvxpy_problem() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(cp.square(y - x)), parameters=[x])
    problem = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        lower,
    )
    assert problem.lower_problem is lower

    raw = cp.Problem(cp.Minimize(cp.square(y - x)))
    with pytest.raises(TypeError, match="BLVPY LowerProblem"):
        BilevelProblem(cp.Minimize(cp.square(x) + cp.square(y)), raw)  # type: ignore[arg-type]


def test_constructor_signature_has_no_parameter_map() -> None:
    parameters = inspect.signature(BilevelProblem).parameters
    assert tuple(parameters) == (
        "upper_objective",
        "lower_problem",
        "upper_constraints",
    )
    assert "parameter_map" not in parameters


def test_solve_defaults_match_documented_continuation_settings() -> None:
    from blvpy.continuation import _SolveSettings

    parameters = inspect.signature(BilevelProblem.solve).parameters
    assert parameters["epsilon_initial"].default == pytest.approx(1e-1)
    assert parameters["epsilon_target"].default == pytest.approx(1e-6)
    assert parameters["contraction"].default == pytest.approx(0.1)
    assert parameters["best_of"].default is None
    assert parameters["feasibility_tolerance"].default == pytest.approx(1e-7)
    assert parameters["solver"].default == cp.IPOPT
    assert parameters["conic_solver"].default == cp.CLARABEL
    assert parameters["verbose"].default is True
    assert parameters["solver_verbose"].default is False
    assert _SolveSettings().verbose is True
    assert _SolveSettings().solver_verbose is False
    assert _SolveSettings().best_of is None


@pytest.mark.parametrize(
    ("verbose", "solver_verbose"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_solve_passes_independent_verbosity_settings(
    monkeypatch: pytest.MonkeyPatch,
    verbose: bool,
    solver_verbose: bool,
) -> None:
    problem, _, _, _ = _quadratic_bilevel()
    captured = None

    def fake_solve_bilevel(model, settings):
        nonlocal captured
        assert model is problem
        captured = settings
        return "result"

    monkeypatch.setattr("blvpy.continuation.solve_bilevel", fake_solve_bilevel)

    result = problem.solve(verbose=verbose, solver_verbose=solver_verbose)

    assert result == "result"
    assert captured is not None
    assert captured.verbose is verbose
    assert captured.solver_verbose is solver_verbose
