from __future__ import annotations

import warnings
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp
from cvxpy.atoms.elementwise.power import PowerApprox
from cvxpy.atoms.geo_mean import GeoMeanApprox
from cvxpy.atoms.pnorm import PnormApprox

from blvpy.canonicalization import (
    CanonicalData,
    CanonicalLowerProblem,
    _canonicalize_lower,
    _symbolic_matrix_combination,
    _symbolic_vector_combination,
)
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


def _expression_nodes(expression: cp.Expression) -> tuple[cp.Expression, ...]:
    nodes: list[cp.Expression] = []
    seen: set[int] = set()
    stack = [expression]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        stack.extend(node.args)
    return tuple(nodes)


def _assert_parameterized_atom_matches_direct(
    problem: cp.Problem,
    parameter: cp.Parameter,
    linked: cp.Variable,
    source: cp.Variable,
    cases: tuple[tuple[float, np.ndarray | float, float], ...],
    *,
    atol: float = 5e-5,
) -> CanonicalLowerProblem:
    """Compare numeric/symbolic canonical data, solves, and recovery."""

    canonical = _canonicalize_lower(problem, {parameter: linked})
    symbolic = canonical.build_data_expressions({parameter: linked})
    packed_data: list[np.ndarray] = []

    for parameter_value, expected_source, expected_objective in cases:
        parameter.value = parameter_value
        linked.value = parameter_value
        direct_data = _direct_data(problem)
        data, primal, canonical_objective = _solve_canonical(canonical, {parameter: parameter_value})
        _assert_data_equal(data, direct_data)
        np.testing.assert_allclose(symbolic.A.value, data.A.toarray())
        np.testing.assert_allclose(symbolic.b.value, data.b)
        np.testing.assert_allclose(symbolic.c.value, data.c)
        assert float(symbolic.d.value) == pytest.approx(data.d)

        direct_objective = problem.solve(solver=cp.CLARABEL)
        assert problem.status in cp.settings.SOLUTION_PRESENT
        recovered = canonical.recover_numeric(primal)[source.id]
        np.testing.assert_allclose(source.value, expected_source, atol=atol)
        np.testing.assert_allclose(recovered, expected_source, atol=atol)
        np.testing.assert_allclose(recovered, source.value, atol=atol)
        assert canonical_objective == pytest.approx(direct_objective, abs=atol)
        assert canonical_objective == pytest.approx(expected_objective, abs=atol)

        packed_data.append(
            np.concatenate(
                [
                    data.A.toarray().reshape(-1),
                    data.b,
                    data.c,
                    np.array([data.d]),
                ]
            )
        )

    assert any(not np.allclose(packed_data[0], item) for item in packed_data[1:])
    return canonical


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


def test_symbolic_affine_helpers_preserve_empty_output_shapes() -> None:
    parameters = cp.Constant(np.array([2.0, 1.0]))
    matrix = _symbolic_matrix_combination(
        (sp.csc_array((0, 3)), sp.csc_array((0, 3))),
        parameters,
        0,
        3,
    )
    vector = _symbolic_vector_combination(np.empty((0, 2)), parameters)

    assert matrix.shape == (0, 3)
    assert np.asarray(matrix.value).shape == (0, 3)
    assert vector.shape == (0,)
    assert np.asarray(vector.value).shape == (0,)


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
    "weights",
    [None, (0.25, 0.25, 0.5)],
    ids=["uniform", "nonuniform"],
)
def test_exact_geometric_mean_matches_direct_cvxpy_at_linked_values(
    weights: tuple[float, ...] | None,
) -> None:
    parameter = cp.Parameter(nonneg=True, name="total", value=1.5)
    linked = cp.Variable(nonneg=True, name="linked_total")
    source = cp.Variable(3, nonneg=True, name="allocation")
    atom = cp.geo_mean(source, p=weights)
    problem = cp.Problem(cp.Minimize(-atom), [cp.sum(source) == parameter])
    normalized_weights = np.full(3, 1.0 / 3.0) if weights is None else np.asarray(weights)
    geometric_factor = float(np.prod(normalized_weights**normalized_weights))

    assert isinstance(atom, GeoMeanApprox)
    assert atom.approx_error == 0.0
    canonical = _assert_parameterized_atom_matches_direct(
        problem,
        parameter,
        linked,
        source,
        (
            (1.5, 1.5 * normalized_weights, -1.5 * geometric_factor),
            (3.0, 3.0 * normalized_weights, -3.0 * geometric_factor),
        ),
        atol=2e-4,
    )

    assert canonical.cone_layout.second_order


@pytest.mark.parametrize(
    "kind",
    ["sqrt", "inv_pos", "convex_rational", "concave_rational"],
)
def test_exact_power_families_match_direct_cvxpy_at_linked_values(kind: str) -> None:
    parameter = cp.Parameter(nonneg=True, name=f"limit_{kind}", value=1.0)
    linked = cp.Variable(nonneg=True, name=f"linked_limit_{kind}")
    source = cp.Variable(name=f"source_{kind}")
    values = (0.75, 2.25)

    if kind == "sqrt":
        atom = cp.sqrt(source)
        objective = -atom
        constraints = [source <= parameter, source >= 0.25]
        expected_objectives = tuple(-np.sqrt(value) for value in values)
    elif kind == "inv_pos":
        atom = cp.inv_pos(source)
        objective = atom
        constraints = [source <= parameter, source >= 0.25]
        expected_objectives = tuple(1.0 / value for value in values)
    elif kind == "convex_rational":
        atom = cp.power(source, Fraction(3, 2))
        objective = atom
        constraints = [source >= parameter]
        expected_objectives = tuple(value**1.5 for value in values)
    else:
        atom = cp.power(source, Fraction(2, 3))
        objective = -atom
        constraints = [source <= parameter, source >= 0.0]
        expected_objectives = tuple(-(value ** (2.0 / 3.0)) for value in values)

    assert isinstance(atom, PowerApprox)
    assert atom.approx_error == 0.0
    canonical = _assert_parameterized_atom_matches_direct(
        cp.Problem(cp.Minimize(objective), constraints),
        parameter,
        linked,
        source,
        tuple(
            (value, value, expected_objective)
            for value, expected_objective in zip(values, expected_objectives, strict=True)
        ),
    )

    assert canonical.cone_layout.second_order


@pytest.mark.parametrize(
    "kind",
    ["convex_pnorm", "concave_pnorm", "harmonic_mean"],
)
def test_exact_rational_pnorm_families_match_direct_cvxpy_at_linked_values(kind: str) -> None:
    parameter = cp.Parameter(nonneg=True, name=f"bound_{kind}", value=1.0)
    linked = cp.Variable(nonneg=True, name=f"linked_bound_{kind}")
    source = cp.Variable(3, nonneg=True, name=f"source_{kind}")
    values = (0.8, 2.0)

    if kind == "convex_pnorm":
        atom = cp.pnorm(source, Fraction(3, 2))
        objective = atom
        constraints = [source[0] >= parameter]
        expected_sources = tuple(np.array([value, 0.0, 0.0]) for value in values)
        expected_objectives = values
    elif kind == "concave_pnorm":
        atom = cp.pnorm(source, Fraction(1, 2))
        objective = -atom
        constraints = [source <= parameter]
        expected_sources = tuple(np.full(3, value) for value in values)
        expected_objectives = tuple(-9.0 * value for value in values)
    else:
        atom = cp.harmonic_mean(source)
        objective = -atom
        constraints = [source <= parameter, source >= 0.25]
        expected_sources = tuple(np.full(3, value) for value in values)
        expected_objectives = tuple(-value for value in values)

    approximation_nodes = [node for node in _expression_nodes(atom) if isinstance(node, PnormApprox)]
    assert len(approximation_nodes) == 1
    assert approximation_nodes[0].approx_error == 0.0
    canonical = _assert_parameterized_atom_matches_direct(
        cp.Problem(cp.Minimize(objective), constraints),
        parameter,
        linked,
        source,
        tuple(
            (value, expected_source, expected_objective)
            for value, expected_source, expected_objective in zip(
                values,
                expected_sources,
                expected_objectives,
                strict=True,
            )
        ),
        atol=8e-5,
    )

    assert canonical.cone_layout.second_order


@pytest.mark.parametrize("axis", [0, 1])
def test_cummax_axes_match_direct_cvxpy_data_and_recovery(axis: int) -> None:
    parameter = cp.Parameter(name=f"shift_axis_{axis}", value=-0.25)
    linked = cp.Variable(name=f"linked_shift_axis_{axis}")
    source = cp.Variable((2, 3), name=f"source_axis_{axis}")
    base = np.array([[1.0, -2.0, 3.0], [0.0, 4.0, -1.0]])
    slope = np.array([[0.5, -0.25, 0.75], [1.0, 0.25, -0.5]])
    target = base + parameter * slope
    atom = cp.cummax(source, axis=axis)
    problem = cp.Problem(cp.Minimize(cp.sum(atom)), [source == target])
    cases = []
    for value in (-0.25, 0.75):
        expected = base + value * slope
        objective = float(np.sum(np.maximum.accumulate(expected, axis=axis)))
        cases.append((value, expected, objective))

    canonical = _assert_parameterized_atom_matches_direct(
        problem,
        parameter,
        linked,
        source,
        tuple(cases),
    )

    assert canonical.cone_layout.second_order == ()


def test_dotsort_matches_direct_cvxpy_data_and_recovery() -> None:
    parameter = cp.Parameter(name="dotsort_shift", value=-0.5)
    linked = cp.Variable(name="linked_dotsort_shift")
    source = cp.Variable(4, name="dotsort_source")
    base = np.array([1.0, -2.0, 3.0, 0.25])
    slope = np.array([0.5, -0.25, 0.75, 1.0])
    weights = np.array([2.0, -1.0, 0.5, 1.25])
    target = base + parameter * slope
    problem = cp.Problem(cp.Minimize(cp.dotsort(source, weights)), [source == target])
    cases = []
    for value in (-0.5, 0.8):
        expected = base + value * slope
        objective = float(np.sort(expected) @ np.sort(weights))
        cases.append((value, expected, objective))

    canonical = _assert_parameterized_atom_matches_direct(
        problem,
        parameter,
        linked,
        source,
        tuple(cases),
    )

    assert canonical.cone_layout.second_order == ()


@pytest.mark.parametrize(
    "problem",
    [
        cp.Problem(cp.Minimize(cp.trace(cp.Variable((2, 2), PSD=True)))),
        cp.Problem(cp.Minimize(cp.exp(cp.Variable()))),
        cp.Problem(cp.Minimize(cp.power(cp.Variable(), 1.5, approx=False))),
        cp.Problem(
            cp.Minimize(-cp.geo_mean(cp.Variable(3, nonneg=True), approx=False)),
        ),
        cp.Problem(
            cp.Minimize(cp.pnorm(cp.Variable(3), Fraction(3, 2), approx=False)),
        ),
    ],
)
def test_unsupported_cones_are_rejected(problem: cp.Problem) -> None:
    with pytest.raises(UnsupportedConeError):
        _canonicalize_lower(problem, {})


@pytest.mark.parametrize("kind", ["power", "pnorm", "geo_mean"])
def test_nonzero_soc_approximation_is_rejected(kind: str) -> None:
    if kind == "power":
        source = cp.Variable(nonneg=True)
        atom = cp.power(source, np.sqrt(2.0))
        problem = cp.Problem(cp.Minimize(atom))
    elif kind == "pnorm":
        source = cp.Variable(3)
        atom = cp.pnorm(source, np.sqrt(2.0))
        problem = cp.Problem(cp.Minimize(atom))
    else:
        source = cp.Variable(2, nonneg=True)
        atom = cp.geo_mean(source, p=[1.0, np.sqrt(2.0)])
        problem = cp.Problem(cp.Minimize(-atom), [cp.sum(source) == 1.0])

    assert atom.approx_error > 0.0
    expected_message = rf"{type(atom).__name__}.*approx_error"
    with pytest.raises(ApproximateCanonicalizationError, match=expected_message):
        _canonicalize_lower(problem, {})


@pytest.mark.parametrize("approximation_error", [np.nan, np.inf, -np.inf, -1e-12])
def test_nonfinite_or_negative_approximation_metadata_is_rejected(approximation_error: float) -> None:
    source = cp.Variable(nonneg=True)
    atom = cp.sqrt(source)
    atom.approx_error = approximation_error
    problem = cp.Problem(cp.Minimize(-atom), [source <= 1.0])

    with pytest.raises(ApproximateCanonicalizationError, match="nonzero or nonfinite approximation error"):
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
