from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from blvpy.canonicalization import CanonicalData, CanonicalLowerProblem, _canonicalize_lower
from blvpy.errors import (
    ApproximateCanonicalizationError,
    ParameterMappingError,
    UnsupportedConeError,
    UnsupportedModelError,
    ValidationError,
)
from blvpy.lower_problem import LowerProblem


def _direct_data(problem: cp.Problem) -> dict:
    data, _, _ = problem.get_problem_data(cp.CLARABEL, enforce_dpp=True, solver_opts={"use_quad_obj": False})
    return data


def _solve_canonical(
    canonical: CanonicalLowerProblem,
    values: dict[cp.Parameter | int, np.ndarray | float],
) -> tuple[CanonicalData, np.ndarray, float]:
    data = canonical.apply_numeric(values)
    primal = cp.Variable(canonical.canonical_size)
    slack = cp.Variable(canonical.constraint_size)
    problem = cp.Problem(
        cp.Minimize(data.c @ primal + data.d),
        [
            data.A @ primal + slack == data.b,
            *canonical.cone_layout.primal_constraints(slack),
        ],
    )
    objective = problem.solve(solver=cp.CLARABEL)
    assert problem.status in cp.settings.SOLUTION_PRESENT
    assert primal.value is not None
    return data, np.asarray(primal.value, dtype=float), float(objective)


def _assert_data_equal(actual: CanonicalData, expected: dict) -> None:
    np.testing.assert_allclose(actual.A.toarray(), expected["A"].toarray())
    np.testing.assert_allclose(actual.b, expected["b"])
    np.testing.assert_allclose(actual.c, expected["c"])
    direct_program = expected["param_prob"]
    _, direct_offset, _, _ = direct_program.apply_parameters()
    assert actual.d == pytest.approx(float(direct_offset))


def _assign_matrix_leaf(leaf: cp.Variable | cp.Parameter, value: np.ndarray) -> None:
    if leaf.attributes["sparsity"]:
        rows, columns = leaf.sparse_idx
        leaf.value_sparse = sp.coo_array(
            (value[rows, columns], (rows, columns)),
            shape=leaf.shape,
        )
    else:
        leaf.value = value


def _matrix_leaf_value(leaf: cp.Variable | cp.Parameter) -> np.ndarray:
    if leaf.attributes["sparsity"]:
        assert leaf.value_sparse is not None
        return np.asarray(leaf.value_sparse.toarray(), dtype=float)
    assert leaf.value is not None
    return np.asarray(leaf.value, dtype=float)


def test_affine_data_matches_cvxpy_for_parameter_dependent_A_b_c() -> None:
    p = cp.Parameter(2, name="p", value=np.array([1.0, 2.0]))
    x = cp.Variable(2, name="x")
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x - p) + p @ x),
        [p[0] * x[0] + x[1] == p[1], x >= p],
    )
    upper = cp.Variable(2)
    canonical = _canonicalize_lower(problem, {p: upper})

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
    canonical = _canonicalize_lower(problem, {parameter: upper})
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


def test_hand_derived_scalar_lp_data_solution_and_recovery() -> None:
    parameter = cp.Parameter(name="parameter", value=0.0)
    source = cp.Variable(name="source")
    problem = cp.Problem(
        cp.Minimize(source + 3.0 * parameter - 2.0),
        [source >= parameter],
    )
    canonical = _canonicalize_lower(problem, {parameter: cp.Variable(name="upper")})

    assert canonical.canonical_size == 1
    assert canonical.constraint_size == 1
    assert canonical.cone_layout.zero == 0
    assert canonical.cone_layout.nonnegative == 1
    assert canonical.cone_layout.second_order == ()

    for value in (-0.7, 0.4, 1.2):
        parameter.value = value
        data, primal, canonical_objective = _solve_canonical(canonical, {parameter: value})

        np.testing.assert_array_equal(data.A.toarray(), [[-1.0]])
        np.testing.assert_array_equal(data.b, [-value])
        np.testing.assert_array_equal(data.c, [1.0])
        assert data.d == pytest.approx(3.0 * value - 2.0)

        recovered = canonical.recover_numeric(primal)[source.id]
        np.testing.assert_allclose(recovered, value, atol=5e-8)
        assert canonical_objective == pytest.approx(4.0 * value - 2.0, abs=5e-8)

        direct_objective = problem.solve(solver=cp.CLARABEL)
        assert problem.status in cp.settings.SOLUTION_PRESENT
        assert source.value == pytest.approx(value, abs=5e-8)
        assert canonical_objective == pytest.approx(direct_objective, abs=5e-8)


@pytest.mark.parametrize(
    ("seed", "dimension"),
    [(3, 2), (17, 3), (41, 4)],
)
def test_seeded_quadratic_halfspace_canonicalization_matches_analytic_projection(
    seed: int,
    dimension: int,
) -> None:
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.15, 0.15, size=dimension)
    slope = rng.uniform(-0.05, 0.05, size=dimension)
    parameter = cp.Parameter(nonneg=True, name=f"parameter_{seed}", value=0.6)
    source = cp.Variable(dimension, name=f"source_{seed}")
    target = base + slope * parameter
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(source - target)),
        [parameter * source[0] + source[1] >= 1.0],
    )
    canonical = _canonicalize_lower(
        problem,
        {parameter: cp.Variable(nonneg=True, name=f"upper_{seed}")},
    )

    for value in (0.6, 1.4):
        parameter.value = value
        target_value = base + slope * value
        normal = np.zeros(dimension)
        normal[:2] = [value, 1.0]
        violation = 1.0 - float(normal @ target_value)
        assert violation > 0.0
        analytic = target_value + violation / float(normal @ normal) * normal
        analytic_objective = float(np.sum(np.square(analytic - target_value)))

        data, primal, canonical_objective = _solve_canonical(canonical, {parameter: value})
        _assert_data_equal(data, _direct_data(problem))
        recovered = canonical.recover_numeric(primal)[source.id]

        direct_objective = problem.solve(solver=cp.CLARABEL)
        assert problem.status in cp.settings.SOLUTION_PRESENT
        np.testing.assert_allclose(source.value, analytic, atol=5e-5)
        np.testing.assert_allclose(recovered, analytic, atol=5e-5)
        np.testing.assert_allclose(recovered, source.value, atol=5e-5)
        assert canonical_objective == pytest.approx(analytic_objective, abs=1e-7)
        assert canonical_objective == pytest.approx(direct_objective, abs=1e-7)


def test_quadratic_objective_becomes_socp_and_recovers_source_variable() -> None:
    parameter = cp.Parameter(2, value=np.array([1.0, -1.0]))
    source = cp.Variable(2, name="source", nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(source - parameter)))
    canonical = _canonicalize_lower(problem, {parameter: cp.Variable(2)})

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
    canonical = _canonicalize_lower(problem, {})
    value = np.arange(canonical.canonical_size, dtype=float)
    recovered = canonical.recover_numeric(value)[source.id]

    assert recovered.shape == (3, 3)
    np.testing.assert_allclose(recovered, recovered.T)


@pytest.mark.parametrize(
    ("kind", "attributes", "values", "transform"),
    [
        (
            "symmetric",
            {"symmetric": True},
            (
                np.array([[1.0, 0.2], [0.2, -0.4]]),
                np.array([[-0.5, 1.1], [1.1, 0.75]]),
            ),
            "symmetric",
        ),
        (
            "diagonal",
            {"diag": True},
            (np.diag([1.0, -2.0]), np.diag([-0.25, 0.8])),
            "diagonal",
        ),
        (
            "sparse",
            {"sparsity": ([0, 1], [1, 0])},
            (
                np.array([[0.0, 1.25], [-0.75, 0.0]]),
                np.array([[0.0, -0.4], [1.5, 0.0]]),
            ),
            "sparse",
        ),
    ],
)
def test_linked_matrix_attribute_packing_matches_direct_cvxpy_solves(
    kind: str,
    attributes: dict,
    values: tuple[np.ndarray, np.ndarray],
    transform: str,
) -> None:
    linked = cp.Variable((2, 2), name=f"linked_{kind}", **attributes)
    source = cp.Variable((2, 2), name=f"source_{kind}")
    offset = np.array([[0.3, -0.2], [0.6, 0.1]])
    lower = LowerProblem(
        cp.Minimize(cp.sum_squares(source - linked - offset)),
        parameters=[linked],
    )
    generated = lower._internal_parameters[0]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Reading from a sparse CVXPY expression via `\.value` is discouraged.*",
            category=RuntimeWarning,
        )
        canonical = _canonicalize_lower(lower._cvxpy_problem, lower._parameter_links)
        spec = canonical.parameter_specs[0]
        assert spec.transform == transform
        assert spec.shape == (2, 2)
        assert spec.internal_size == {"symmetric": 3, "diagonal": 2, "sparse": 2}[kind]

        for value in values:
            _assign_matrix_leaf(linked, value)
            _assign_matrix_leaf(generated, value)
            np.testing.assert_allclose(_matrix_leaf_value(generated), value)

            expected_packed = {
                "symmetric": value[np.triu_indices(2)],
                "diagonal": np.diag(value),
                "sparse": value[generated.sparse_idx],
            }[kind]
            np.testing.assert_allclose(spec.pack_numeric(value), expected_packed)

            direct_data = _direct_data(lower._cvxpy_problem)
            data, primal, canonical_objective = _solve_canonical(canonical, {generated: value})
            _assert_data_equal(data, direct_data)
            recovered = canonical.recover_numeric(primal)[source.id]

            direct_objective = lower._cvxpy_problem.solve(solver=cp.CLARABEL)
            assert lower._cvxpy_problem.status in cp.settings.SOLUTION_PRESENT
            expected_source = value + offset
            np.testing.assert_allclose(source.value, expected_source, atol=5e-5)
            np.testing.assert_allclose(recovered, expected_source, atol=5e-5)
            np.testing.assert_allclose(recovered, source.value, atol=5e-5)
            assert canonical_objective == pytest.approx(direct_objective, abs=1e-7)
            assert canonical_objective == pytest.approx(0.0, abs=1e-7)
            assert lower.objective.value == pytest.approx(direct_objective, abs=1e-9)


def test_recovery_agrees_with_direct_solve_for_multiple_shapes_and_attributes() -> None:
    parameter = cp.Parameter(name="parameter", value=0.4)
    scalar = cp.Variable(name="scalar")
    vector = cp.Variable(2, name="vector")
    matrix = cp.Variable((2, 2), name="matrix")
    nonnegative = cp.Variable(2, nonneg=True, name="nonnegative")
    bounded = cp.Variable(2, bounds=[-0.5, 2.0], name="bounded")
    matrix_target = cp.vstack([cp.hstack([parameter, 1.0]), cp.hstack([2.0, -parameter])])
    problem = cp.Problem(
        cp.Minimize(
            cp.square(scalar - (parameter + 1.0))
            + cp.sum_squares(vector - cp.hstack([parameter, -parameter]))
            + cp.sum_squares(matrix - matrix_target)
            + cp.sum_squares(nonnegative - cp.hstack([parameter + 2.0, 1.0]))
            + cp.sum_squares(bounded - cp.hstack([0.5 * parameter, 1.5]))
        )
    )
    canonical = _canonicalize_lower(problem, {parameter: cp.Variable()})
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


def test_recovery_agrees_with_attribute_boundary_optima() -> None:
    symmetric = cp.Variable((2, 2), symmetric=True, name="symmetric")
    nonnegative = cp.Variable(2, nonneg=True, name="nonnegative")
    bounded = cp.Variable(2, bounds=[-0.5, 2.0], name="bounded")
    symmetric_target = np.array([[1.2, -0.35], [-0.35, -0.6]])
    problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(symmetric - symmetric_target)
            + cp.sum_squares(nonnegative - np.array([-1.0, -0.25]))
            + cp.sum_squares(bounded - np.array([3.0, 2.5]))
        )
    )
    canonical = _canonicalize_lower(problem, {})
    _, primal, canonical_objective = _solve_canonical(canonical, {})
    recovered = canonical.recover_numeric(primal)

    direct_objective = problem.solve(solver=cp.CLARABEL)
    assert problem.status in cp.settings.SOLUTION_PRESENT
    expected_nonnegative = np.zeros(2)
    expected_bounded = np.full(2, 2.0)

    np.testing.assert_allclose(symmetric.value, symmetric_target, atol=5e-5)
    np.testing.assert_allclose(recovered[symmetric.id], symmetric_target, atol=5e-5)
    np.testing.assert_allclose(recovered[symmetric.id], recovered[symmetric.id].T, atol=1e-12)
    np.testing.assert_allclose(nonnegative.value, expected_nonnegative, atol=5e-5)
    np.testing.assert_allclose(recovered[nonnegative.id], expected_nonnegative, atol=5e-5)
    np.testing.assert_allclose(bounded.value, expected_bounded, atol=5e-5)
    np.testing.assert_allclose(recovered[bounded.id], expected_bounded, atol=5e-5)
    assert canonical_objective == pytest.approx(direct_objective, abs=1e-7)


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
        _canonicalize_lower(problem, {})


def test_only_genuine_soc_approximation_is_rejected() -> None:
    x = cp.Variable()
    with pytest.raises(ApproximateCanonicalizationError):
        _canonicalize_lower(cp.Problem(cp.Minimize(cp.power(x, 1.23456789))), {})

    exact_square = _canonicalize_lower(cp.Problem(cp.Minimize(cp.square(x))), {})
    assert exact_square.cone_layout.soc == (3,)


def test_exact_but_unaudited_socp_atom_is_rejected() -> None:
    source = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Minimize(-cp.geo_mean(source)), [cp.sum(source) == 1.0])

    direct = _direct_data(problem)
    assert direct["dims"].soc
    with pytest.raises(UnsupportedModelError, match="audited exact SOCP.*allowlist"):
        _canonicalize_lower(problem, {})


def test_constant_only_lower_problem_has_explicit_rejection() -> None:
    problem = cp.Problem(cp.Minimize(cp.Constant(3.0)))

    with pytest.raises(UnsupportedModelError, match="constant-only lower problems"):
        _canonicalize_lower(problem, {})


def test_invalid_source_models_are_rejected() -> None:
    x = cp.Variable()
    parameter = cp.Parameter(2)
    fixed_missing = cp.Problem(cp.Minimize(cp.sum_squares(x - parameter[0])))
    with pytest.raises(ParameterMappingError, match="fixed value"):
        _canonicalize_lower(fixed_missing, {})
    with pytest.raises(ValidationError, match="minimization"):
        _canonicalize_lower(cp.Problem(cp.Maximize(x)), {})

    integer = cp.Variable(integer=True)
    with pytest.raises(UnsupportedModelError, match="Mixed-integer"):
        _canonicalize_lower(cp.Problem(cp.Minimize(integer)), {})

    complex_variable = cp.Variable(complex=True)
    with pytest.raises(UnsupportedModelError, match="Complex"):
        _canonicalize_lower(cp.Problem(cp.Minimize(cp.square(cp.real(complex_variable)))), {})


def test_dcp_but_non_dpp_lower_problem_is_rejected() -> None:
    parameter = cp.Parameter(pos=True, value=1.0)
    source = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.quad_over_lin(source, parameter)))

    assert problem.is_dcp()
    assert not problem.is_dpp()
    with pytest.raises(ValidationError, match="DPP"):
        _canonicalize_lower(problem, {parameter: cp.Variable(pos=True)})


def test_dpp_is_checked_only_with_respect_to_mapped_parameters() -> None:
    mapped = cp.Parameter(name="mapped")
    fixed = cp.Parameter(pos=True, value=2.0, name="fixed")
    source = cp.Variable(name="source")
    problem = cp.Problem(cp.Minimize(cp.quad_over_lin(source, fixed) + cp.square(source - mapped)))

    assert not problem.is_dpp()
    canonical = _canonicalize_lower(problem, {mapped: cp.Variable()})
    assert canonical.parameter_ids == (mapped.id,)
    assert fixed.id in canonical.fixed_parameter_values


def test_quadrature_approximation_is_rejected_before_cone_solving() -> None:
    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonneg=True)
    z = cp.Variable()
    problem = cp.Problem(
        cp.Minimize(z),
        [cp.RelEntrConeQuad(x, y, z, m=3, k=2)],
    )

    with pytest.raises(ApproximateCanonicalizationError, match="quadrature"):
        _canonicalize_lower(problem, {})
