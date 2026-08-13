from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from blvpy.canonicalization import canonicalize_lower
from blvpy.errors import (
    ApproximateCanonicalizationError,
    ParameterMappingError,
    UnsupportedConeError,
    UnsupportedModelError,
    ValidationError,
)


def _direct_data(problem: cp.Problem) -> dict:
    data, _, _ = problem.get_problem_data(
        cp.CLARABEL, enforce_dpp=True, solver_opts={"use_quad_obj": False}
    )
    return data


def test_affine_data_matches_cvxpy_for_parameter_dependent_A_b_c() -> None:
    p = cp.Parameter(2, name="p", value=np.array([1.0, 2.0]))
    x = cp.Variable(2, name="x")
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x - p) + p @ x),
        [p[0] * x[0] + x[1] == p[1], x >= p],
    )
    upper = cp.Variable(2)
    canonical = canonicalize_lower(problem, {p: upper})

    for value in (np.array([2.0, 3.0]), np.array([-0.5, 4.0])):
        actual = canonical.apply_numeric({p.id: value})
        p.value = value
        expected = _direct_data(problem)
        np.testing.assert_allclose(actual.A.toarray(), expected["A"].toarray())
        np.testing.assert_allclose(actual.b, expected["b"])
        np.testing.assert_allclose(actual.c, expected["c"])

    symbolic = canonical.build_data_expressions({p.id: upper})
    upper.value = np.array([0.25, 1.5])
    p.value = upper.value
    expected = _direct_data(problem)
    np.testing.assert_allclose(symbolic.A.value, expected["A"].toarray())
    np.testing.assert_allclose(symbolic.b.value, expected["b"])
    np.testing.assert_allclose(symbolic.c.value, expected["c"])


def test_affine_objective_offset_matches_cvxpy_at_multiple_values() -> None:
    parameter = cp.Parameter(name="parameter", value=0.25)
    source = cp.Variable(name="source")
    upper = cp.Variable(name="upper")
    problem = cp.Problem(
        cp.Minimize(parameter * source + 3.0 * parameter - 2.0),
        [source >= 1.0, source <= 4.0],
    )
    canonical = canonicalize_lower(problem, {parameter: upper})
    symbolic = canonical.build_data_expressions({parameter: upper})

    for value in (-2.0, 0.0, 1.75):
        parameter.value = value
        upper.value = value
        direct_program = _direct_data(problem)["param_prob"]
        _, direct_offset, _, _ = direct_program.apply_parameters()
        actual = canonical.apply_numeric({parameter: value})

        assert actual.d == pytest.approx(float(direct_offset))
        assert actual.d == pytest.approx(3.0 * value - 2.0)
        assert float(symbolic.d.value) == pytest.approx(actual.d)


def test_quadratic_objective_becomes_socp_and_recovers_source_variable() -> None:
    parameter = cp.Parameter(2, value=np.array([1.0, -1.0]))
    source = cp.Variable(2, name="source", nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(source - parameter)))
    canonical = canonicalize_lower(problem, {parameter: cp.Variable(2)})

    assert canonical.cone_layout.soc
    assert canonical.source_variable_ids == (source.id,)
    canonical_value = np.arange(canonical.canonical_size, dtype=float)
    recovered = canonical.recover_numeric(canonical_value)[source.id]
    expression = canonical.recovery_expressions(cp.Constant(canonical_value))[source.id]
    np.testing.assert_allclose(expression.value, recovered)
    np.testing.assert_allclose(
        recovered,
        canonical.recovery_specs[0].matrix @ canonical_value,
    )


def test_symmetric_source_recovery_is_affine_and_shaped() -> None:
    source = cp.Variable((3, 3), symmetric=True, name="matrix")
    problem = cp.Problem(cp.Minimize(cp.sum_squares(source)))
    canonical = canonicalize_lower(problem, {})
    value = np.arange(canonical.canonical_size, dtype=float)
    recovered = canonical.recover_numeric(value)[source.id]

    assert recovered.shape == (3, 3)
    np.testing.assert_allclose(recovered, recovered.T)


def test_recovery_agrees_with_direct_solve_for_multiple_shapes_and_attributes() -> None:
    parameter = cp.Parameter(name="parameter", value=0.4)
    scalar = cp.Variable(name="scalar")
    vector = cp.Variable(2, name="vector")
    matrix = cp.Variable((2, 2), name="matrix")
    nonnegative = cp.Variable(2, nonneg=True, name="nonnegative")
    bounded = cp.Variable(2, bounds=[-0.5, 2.0], name="bounded")
    matrix_target = cp.vstack(
        [cp.hstack([parameter, 1.0]), cp.hstack([2.0, -parameter])]
    )
    problem = cp.Problem(
        cp.Minimize(
            cp.square(scalar - (parameter + 1.0))
            + cp.sum_squares(vector - cp.hstack([parameter, -parameter]))
            + cp.sum_squares(matrix - matrix_target)
            + cp.sum_squares(nonnegative - cp.hstack([parameter + 2.0, 1.0]))
            + cp.sum_squares(bounded - cp.hstack([0.5 * parameter, 1.5]))
        )
    )
    canonical = canonicalize_lower(problem, {parameter: cp.Variable()})
    data = canonical.apply_numeric({parameter: parameter.value})
    primal = cp.Variable(canonical.canonical_size)
    slack = cp.Variable(canonical.constraint_size)
    canonical_problem = cp.Problem(
        cp.Minimize(data.c @ primal + data.d),
        [
            data.A @ primal + slack == data.b,
            *canonical.cone_layout.primal_constraints(slack),
        ],
    )
    canonical_problem.solve(solver=cp.CLARABEL)
    assert canonical_problem.status in cp.settings.SOLUTION_PRESENT
    recovered = canonical.recover_numeric(primal.value)

    problem.solve(solver=cp.CLARABEL)
    assert problem.status in cp.settings.SOLUTION_PRESENT
    for source in (scalar, vector, matrix, nonnegative, bounded):
        assert recovered[source.id].shape == source.shape
        np.testing.assert_allclose(recovered[source.id], source.value, atol=5e-5)


@pytest.mark.parametrize(
    "problem",
    [
        cp.Problem(cp.Minimize(cp.trace(cp.Variable((2, 2), PSD=True)))),
        cp.Problem(cp.Minimize(cp.exp(cp.Variable()))),
        cp.Problem(cp.Minimize(cp.power(cp.Variable(), 1.5, approx=False))),
    ],
)
def test_unsupported_cones_are_rejected(problem: cp.Problem) -> None:
    with pytest.raises(UnsupportedConeError):
        canonicalize_lower(problem, {})


def test_only_genuine_soc_approximation_is_rejected() -> None:
    x = cp.Variable()
    with pytest.raises(ApproximateCanonicalizationError):
        canonicalize_lower(cp.Problem(cp.Minimize(cp.power(x, 1.23456789))), {})

    exact_square = canonicalize_lower(cp.Problem(cp.Minimize(cp.square(x))), {})
    assert exact_square.cone_layout.soc == (3,)


def test_exact_but_unaudited_socp_atom_is_rejected() -> None:
    source = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Minimize(-cp.geo_mean(source)), [cp.sum(source) == 1.0])

    direct = _direct_data(problem)
    assert direct["dims"].soc
    with pytest.raises(UnsupportedModelError, match="audited exact SOCP.*allowlist"):
        canonicalize_lower(problem, {})


def test_constant_only_lower_problem_has_explicit_rejection() -> None:
    problem = cp.Problem(cp.Minimize(cp.Constant(3.0)))

    with pytest.raises(UnsupportedModelError, match="constant-only lower problems"):
        canonicalize_lower(problem, {})


def test_invalid_source_models_and_parameter_mappings_are_rejected() -> None:
    x = cp.Variable()
    parameter = cp.Parameter(2)
    fixed_missing = cp.Problem(cp.Minimize(cp.sum_squares(x - parameter[0])))
    with pytest.raises(ParameterMappingError, match="fixed value"):
        canonicalize_lower(fixed_missing, {})
    with pytest.raises(ParameterMappingError, match="shape"):
        canonicalize_lower(fixed_missing, {parameter: cp.Variable(3)})
    with pytest.raises(ValidationError, match="minimization"):
        canonicalize_lower(cp.Problem(cp.Maximize(x)), {})

    integer = cp.Variable(integer=True)
    with pytest.raises(UnsupportedModelError, match="Mixed-integer"):
        canonicalize_lower(cp.Problem(cp.Minimize(integer)), {})

    complex_variable = cp.Variable(complex=True)
    with pytest.raises(UnsupportedModelError, match="Complex"):
        canonicalize_lower(cp.Problem(cp.Minimize(cp.square(cp.real(complex_variable)))), {})


def test_dcp_but_non_dpp_lower_problem_is_rejected() -> None:
    parameter = cp.Parameter(pos=True, value=1.0)
    source = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.quad_over_lin(source, parameter)))

    assert problem.is_dcp()
    assert not problem.is_dpp()
    with pytest.raises(ValidationError, match="DPP"):
        canonicalize_lower(problem, {parameter: cp.Variable(pos=True)})


def test_dpp_is_checked_only_with_respect_to_mapped_parameters() -> None:
    mapped = cp.Parameter(name="mapped")
    fixed = cp.Parameter(pos=True, value=2.0, name="fixed")
    source = cp.Variable(name="source")
    problem = cp.Problem(
        cp.Minimize(cp.quad_over_lin(source, fixed) + cp.square(source - mapped))
    )

    assert not problem.is_dpp()
    canonical = canonicalize_lower(problem, {mapped: cp.Variable()})
    assert canonical.parameter_ids == (mapped.id,)
    assert all(parameter.id != fixed.id for parameter in canonical.canonical_problem.parameters())


def test_quadrature_approximation_is_rejected_before_cone_solving() -> None:
    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonneg=True)
    z = cp.Variable()
    problem = cp.Problem(
        cp.Minimize(z),
        [cp.RelEntrConeQuad(x, y, z, m=3, k=2)],
    )

    with pytest.raises(ApproximateCanonicalizationError, match="quadrature"):
        canonicalize_lower(problem, {})
