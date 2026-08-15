"""Native IPOPT end-to-end checks against independent numerical oracles.

These tests are marked because they execute against the native IPOPT library.
Every returned bilevel point is checked without using BLVPY's residual helper,
and its fixed-upper lower problem is solved again with Clarabel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from blvpy import BilevelProblem, GapDiagnostics, LowerProblem, Residuals

if TYPE_CHECKING:
    from blvpy import BilevelResult, ConeLayout

pytestmark = pytest.mark.ipopt

_ANALYTIC_ATOL = 3e-3
_OBJECTIVE_ATOL = 5e-3
_FINAL_RESIDUAL_TOL = 1e-5
_SOURCE_GAP_LOWER_TOL = 1e-6
_SOURCE_GAP_UPPER_TOL = 1e-6
_SOLVER_OPTIONS = {
    "hessian_approximation": "limited-memory",
    "tol": 1e-8,
}


@dataclass(frozen=True)
class _NumericalOracle:
    source_gap: float
    direct_lower_value: float
    residuals: Residuals
    gap_diagnostics: GapDiagnostics


def _snapshot_source_values(
    model: BilevelProblem,
    result: BilevelResult,
) -> dict[cp.Variable, NDArray[np.float64]]:
    assert set(result.variable_values) == set(model.source_variables)
    return {
        variable: np.array(result.variable_values[variable], dtype=float, copy=True)
        for variable in model.source_variables
    }


def _assign_source_values(values: dict[cp.Variable, NDArray[np.float64]]) -> None:
    for variable, value in values.items():
        variable.project_and_assign(value)


def _soc_distance(point: ArrayLike) -> float:
    vector = np.asarray(point, dtype=float).reshape(-1)
    head = float(vector[0])
    tail_norm = float(np.linalg.norm(vector[1:]))
    if tail_norm <= head:
        return 0.0
    if tail_norm <= -head:
        return float(np.linalg.norm(vector))
    return (tail_norm - head) / sqrt(2.0)


def _product_cone_distance(
    value: ArrayLike,
    layout: ConeLayout,
    *,
    dual: bool,
) -> float:
    vector = np.asarray(value, dtype=float).reshape(-1)
    squared = 0.0
    if not dual:
        squared += float(vector[layout.zero_slice] @ vector[layout.zero_slice])
    nonnegative = vector[layout.nonnegative_slice]
    negative_part = np.minimum(nonnegative, 0.0)
    squared += float(negative_part @ negative_part)
    for block in layout.second_order_slices:
        squared += _soc_distance(vector[block]) ** 2
    return sqrt(squared)


def _upper_constraint_violation(model: BilevelProblem) -> float:
    violation = 0.0
    for constraint in model._lifted_problem.upper_constraints:
        value = np.asarray(constraint.violation(), dtype=float).reshape(-1)
        violation = max(violation, float(np.linalg.norm(value)))
    return violation


def _independent_residuals(
    model: BilevelProblem,
    result: BilevelResult,
    source_values: dict[cp.Variable, NDArray[np.float64]],
) -> tuple[Residuals, GapDiagnostics]:
    assert result.canonical_primal is not None
    assert result.slack is not None
    assert result.dual is not None
    assert result.final_epsilon is not None

    parameter_values = {parameter: source_values[variable] for parameter, variable in model._parameter_links.items()}
    canonical = model.canonicalize()
    data = canonical.apply_numeric(parameter_values)
    primal = np.asarray(result.canonical_primal, dtype=float).reshape(-1)
    slack = np.asarray(result.slack, dtype=float).reshape(-1)
    dual = np.asarray(result.dual, dtype=float).reshape(-1)
    primal_residual = np.asarray(data.A @ primal + slack - data.b).reshape(-1)
    dual_residual = np.asarray(data.A.T @ dual + data.c).reshape(-1)

    recovered = {
        spec.variable_id: np.asarray(spec.matrix @ primal + spec.offset).reshape(spec.shape, order="F")
        for spec in canonical.recovery_specs
    }
    recovery = max(
        (
            float(np.linalg.norm(source_values[variable] - recovered[variable.id]))
            for variable in model._cvxpy_lower_problem.variables()
        ),
        default=0.0,
    )
    complementarity = float(slack @ dual)
    layout = canonical.cone_layout
    residuals = Residuals(
        primal_equality=float(np.linalg.norm(primal_residual)),
        dual_equality=float(np.linalg.norm(dual_residual)),
        recovery=recovery,
        upper_constraints=_upper_constraint_violation(model),
        primal_cone=_product_cone_distance(slack, layout, dual=False),
        dual_cone=_product_cone_distance(dual, layout, dual=True),
        complementarity=complementarity,
        gap_violation=max(complementarity - result.final_epsilon, 0.0),
    )
    diagnostics = GapDiagnostics(
        primal_objective=float(data.c @ primal),
        dual_objective=float(-(data.b @ dual)),
        complementarity=complementarity,
        dual_residual_term=float(primal @ dual_residual),
        primal_residual_term=float(dual @ primal_residual),
    )
    return residuals, diagnostics


def _primal_cone_constraints(value: cp.Variable, layout: ConeLayout) -> list[cp.Constraint]:
    constraints: list[cp.Constraint] = []
    if layout.zero:
        constraints.append(value[layout.zero_slice] == 0.0)
    if layout.nonnegative:
        constraints.append(value[layout.nonnegative_slice] >= 0.0)
    constraints.extend(
        cp.SOC(value[block.start], value[block.start + 1 : block.stop])
        for block in layout.blocks
        if block.kind == "second_order"
    )
    return constraints


def _fresh_fixed_lower_reference(
    model: BilevelProblem,
    source_values: dict[cp.Variable, NDArray[np.float64]],
    *,
    check_recovery: bool,
) -> float:
    parameter_values = {parameter: source_values[variable] for parameter, variable in model._parameter_links.items()}
    for parameter, value in parameter_values.items():
        parameter.value = value

    generated = model._cvxpy_lower_problem
    source_reference = cp.Problem(generated.objective, list(generated.constraints))
    source_reference.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_feas=1e-10,
    )
    assert source_reference.status in cp.settings.SOLUTION_PRESENT
    direct_lower_value = float(source_reference.value)
    direct_source_values = {
        variable.id: np.array(variable.value, dtype=float, copy=True) for variable in generated.variables()
    }

    canonical = model.canonicalize()
    data = canonical.apply_numeric(parameter_values)
    primal = cp.Variable(canonical.canonical_size, name="reference_primal")
    slack = cp.Variable(canonical.constraint_size, name="reference_slack")
    canonical_reference = cp.Problem(
        cp.Minimize(data.c @ primal + data.d),
        [data.A @ primal + slack == data.b, *_primal_cone_constraints(slack, canonical.cone_layout)],
    )
    canonical_reference.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_feas=1e-10,
    )
    assert canonical_reference.status in cp.settings.SOLUTION_PRESENT
    assert canonical_reference.value == pytest.approx(direct_lower_value, abs=1e-8)

    if check_recovery:
        canonical_primal = np.asarray(primal.value, dtype=float).reshape(-1)
        for spec in canonical.recovery_specs:
            recovered = np.asarray(spec.matrix @ canonical_primal + spec.offset).reshape(spec.shape, order="F")
            np.testing.assert_allclose(recovered, direct_source_values[spec.variable_id], atol=5e-5)
    return direct_lower_value


def _assert_residuals_match(actual: Residuals, expected: Residuals) -> None:
    names = (
        "primal_equality",
        "dual_equality",
        "recovery",
        "upper_constraints",
        "primal_cone",
        "dual_cone",
        "complementarity",
        "gap_violation",
    )
    np.testing.assert_allclose(
        [getattr(actual, name) for name in names],
        [getattr(expected, name) for name in names],
        rtol=1e-8,
        atol=1e-10,
    )


def _assert_gap_diagnostics_match(actual: GapDiagnostics, expected: GapDiagnostics) -> None:
    names = (
        "primal_objective",
        "dual_objective",
        "complementarity",
        "dual_residual_term",
        "primal_residual_term",
        "normalized_gap",
        "inexact_identity_rhs",
        "identity_error",
    )
    np.testing.assert_allclose(
        [getattr(actual, name) for name in names],
        [getattr(expected, name) for name in names],
        rtol=1e-8,
        atol=1e-9,
    )
    assert actual.source_gap == pytest.approx(expected.source_gap, abs=1e-8)


def _check_against_numerical_oracles(
    model: BilevelProblem,
    result: BilevelResult,
    *,
    check_reference_recovery: bool = True,
    check_gap_convenience: bool = False,
) -> _NumericalOracle:
    assert result.succeeded
    assert result.objective is not None
    assert result.residuals is not None
    assert result.final_epsilon is not None

    source_values = _snapshot_source_values(model, result)
    _assign_source_values(source_values)
    evaluated_outer_objective = float(model.outer_objective.value)
    assert result.objective == pytest.approx(evaluated_outer_objective, abs=1e-9)
    returned_lower_value = float(model.lower_problem.objective.expr.value)

    parameter_values = {parameter: source_values[variable] for parameter, variable in model._parameter_links.items()}
    data = model.canonicalize().apply_numeric(parameter_values)
    canonical_primal = np.asarray(result.canonical_primal, dtype=float).reshape(-1)
    returned_canonical_value = float(data.c @ canonical_primal + data.d)
    assert returned_canonical_value == pytest.approx(returned_lower_value, abs=1e-8)

    independent_residuals, gap_diagnostics = _independent_residuals(
        model,
        result,
        source_values,
    )
    _assert_residuals_match(result.residuals, independent_residuals)
    assert independent_residuals.max_violation <= _FINAL_RESIDUAL_TOL

    try:
        direct_lower_value = _fresh_fixed_lower_reference(
            model,
            source_values,
            check_recovery=check_reference_recovery,
        )
    finally:
        _assign_source_values(source_values)

    source_gap = returned_lower_value - direct_lower_value
    assert source_gap >= -_SOURCE_GAP_LOWER_TOL
    assert source_gap <= result.final_epsilon + _SOURCE_GAP_UPPER_TOL

    gap_diagnostics = GapDiagnostics(
        primal_objective=gap_diagnostics.primal_objective,
        dual_objective=gap_diagnostics.dual_objective,
        complementarity=gap_diagnostics.complementarity,
        dual_residual_term=gap_diagnostics.dual_residual_term,
        primal_residual_term=gap_diagnostics.primal_residual_term,
        source_gap=source_gap,
    )
    assert gap_diagnostics.source_gap == pytest.approx(source_gap)
    assert gap_diagnostics.identity_error == pytest.approx(0.0, abs=1e-9)
    if check_gap_convenience:
        _assert_gap_diagnostics_match(model.gap_diagnostics(result), gap_diagnostics)
    return _NumericalOracle(
        source_gap=source_gap,
        direct_lower_value=direct_lower_value,
        residuals=independent_residuals,
        gap_diagnostics=gap_diagnostics,
    )


def _solve(
    model: BilevelProblem,
    *,
    epsilon_initial: float = 1e-2,
    epsilon_target: float = 1e-5,
    best_of: int | None = None,
    seed: int = 0,
) -> BilevelResult:
    return model.solve(
        epsilon_initial=epsilon_initial,
        epsilon_target=epsilon_target,
        best_of=best_of,
        seed=seed,
        solver_options=_SOLVER_OPTIONS,
    )


def _quadratic_model() -> tuple[BilevelProblem, cp.Variable, cp.Variable]:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(
        cp.Minimize(cp.square(y - x) + 2.0 * x - 1.0),
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
        lower,
    )
    return model, x, y


def test_analytic_quadratic_reaches_target_and_is_epsilon_lower_optimal() -> None:
    model, x, y = _quadratic_model()

    result = _solve(model, seed=4)

    assert result.final_epsilon == pytest.approx(1e-5)
    assert all(left > right for left, right in zip(result.epsilon_history, result.epsilon_history[1:]))
    displacement = sqrt(result.final_epsilon) / 2.0
    np.testing.assert_allclose([x.value, y.value], [displacement, -displacement], atol=1e-3)
    expected_objective = 2.0 - 2.0 * sqrt(result.final_epsilon) + result.final_epsilon / 2.0
    assert result.objective == pytest.approx(expected_objective, abs=2e-3)

    parameter = next(iter(model._parameter_links))
    low = model.canonicalize().apply_numeric({parameter: -0.25})
    high = model.canonicalize().apply_numeric({parameter: 0.75})
    assert high.d - low.d == pytest.approx(2.0)

    oracle = _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
    )
    assert oracle.source_gap == pytest.approx(result.final_epsilon, abs=3e-6)


def test_optimistic_lp_selects_upper_preferred_lower_optimizer() -> None:
    x = cp.Variable(name="x", bounds=[0.0, 1.0])
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(0.0 * y), [y >= x, y <= 1.0], parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
        lower,
    )

    result = _solve(model, seed=11)

    np.testing.assert_allclose([x.value, y.value], [0.0, 1.0], atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    oracle = _check_against_numerical_oracles(
        model,
        result,
        check_reference_recovery=False,
        check_gap_convenience=True,
    )
    assert oracle.source_gap == pytest.approx(0.0, abs=1e-8)


def test_parameter_dependent_socp_with_active_upper_constraint() -> None:
    x = cp.Variable(name="x", bounds=[0.25, 1.5])
    y = cp.Variable(2, name="y")
    t = cp.Variable(name="t")
    lower = LowerProblem(
        cp.Minimize(t + x * y[0]),
        [cp.SOC(t, y), y[0] == 1.0, x * y[0] + y[1] == 1.0],
        parameters=[x],
    )
    outer_constraint = x <= 1.0
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 2.0) + cp.sum_squares(y - np.array([1.0, 0.0])) + cp.square(t - 1.0)),
        lower,
        outer_constraints=[outer_constraint],
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.zero == 2
    assert canonical.cone_layout.nonnegative == 0
    assert canonical.cone_layout.second_order == (3,)
    parameter = next(iter(model._parameter_links))
    low = canonical.apply_numeric({parameter: 0.5})
    high = canonical.apply_numeric({parameter: 1.0})
    assert not np.allclose(low.A.toarray(), high.A.toarray())
    assert not np.allclose(low.c, high.c)

    result = _solve(model, seed=7)

    assert float(x.value) == pytest.approx(1.0, abs=_ANALYTIC_ATOL)
    np.testing.assert_allclose(y.value, [1.0, 0.0], atol=_ANALYTIC_ATOL)
    assert float(t.value) == pytest.approx(1.0, abs=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(1.0, abs=_OBJECTIVE_ATOL)
    assert float(np.asarray(outer_constraint.violation())) <= 1e-7
    assert abs(float(x.value) - 1.0) <= _ANALYTIC_ATOL
    assert abs(float(t.value) - np.linalg.norm(y.value)) <= _ANALYTIC_ATOL
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
    )


def test_vector_matrix_links_multiple_soc_blocks_and_fixed_parameter() -> None:
    vector = cp.Variable(2, name="vector", bounds=[-2.0, 2.0])
    matrix = cp.Variable((2, 2), name="matrix", bounds=[-2.0, 2.0])
    lower_vector = cp.Variable(2, name="lower_vector")
    lower_matrix = cp.Variable((2, 2), name="lower_matrix")
    vector_epigraph = cp.Variable(name="vector_epigraph")
    matrix_epigraph = cp.Variable(name="matrix_epigraph")
    fixed_weight = cp.Parameter(nonneg=True, name="fixed_weight", value=2.0)
    lower = LowerProblem(
        cp.Minimize(fixed_weight * (vector_epigraph + matrix_epigraph)),
        [
            cp.SOC(vector_epigraph, lower_vector - vector),
            lower_vector[0] >= vector[0] + 1.0,
            cp.SOC(matrix_epigraph, cp.vec(lower_matrix - matrix, order="F")),
            lower_matrix[0, 0] >= matrix[0, 0] + 1.0,
        ],
        parameters=[vector, matrix],
    )
    target_vector = np.array([0.25, -0.5])
    target_matrix = np.array([[0.5, -0.25], [0.75, 1.0]])
    target_lower_vector = target_vector + np.array([1.0, 0.0])
    target_lower_matrix = target_matrix + np.array([[1.0, 0.0], [0.0, 0.0]])
    model = BilevelProblem(
        cp.Minimize(
            cp.sum_squares(vector - target_vector)
            + cp.sum_squares(matrix - target_matrix)
            + cp.sum_squares(lower_vector - target_lower_vector)
            + cp.sum_squares(lower_matrix - target_lower_matrix)
            + cp.square(vector_epigraph - 1.0)
            + cp.square(matrix_epigraph - 1.0)
        ),
        lower,
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.zero == 0
    assert canonical.cone_layout.nonnegative == 2
    assert canonical.cone_layout.second_order == (3, 5)
    assert fixed_weight.id in canonical.fixed_parameter_values

    result = _solve(model, seed=23)

    np.testing.assert_allclose(vector.value, target_vector, atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(matrix.value, target_matrix, atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(lower_vector.value, target_lower_vector, atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(lower_matrix.value, target_lower_matrix, atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose([vector_epigraph.value, matrix_epigraph.value], [1.0, 1.0], atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    assert result.variable_values[vector].shape == (2,)
    assert result.variable_values[matrix].shape == (2, 2)
    assert result.variable_values[lower_vector].shape == (2,)
    assert result.variable_values[lower_matrix].shape == (2, 2)
    _check_against_numerical_oracles(model, result)


def test_scalar_lp_returns_hand_derived_kkt_point_and_canonical_data() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(cp.Minimize(y + 3.0 * x - 2.0), [y >= x], parameters=[x])
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 0.4) + cp.square(y - 0.4)),
        lower,
    )

    result = _solve(model, seed=29)

    np.testing.assert_allclose([x.value, y.value], [0.4, 0.4], atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    np.testing.assert_allclose(result.canonical_primal, [0.4], atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(result.slack, [0.0], atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(result.dual, [1.0], atol=_ANALYTIC_ATOL)

    parameter = next(iter(model._parameter_links))
    data = model.canonicalize().apply_numeric({parameter: float(x.value)})
    np.testing.assert_allclose(data.A.toarray(), [[-1.0]], atol=1e-12)
    np.testing.assert_allclose(data.b, [-float(x.value)], atol=1e-12)
    np.testing.assert_allclose(data.c, [1.0], atol=1e-12)
    assert data.d == pytest.approx(3.0 * float(x.value) - 2.0)
    _check_against_numerical_oracles(model, result)


@pytest.mark.parametrize("epsilon_target", [1e-1, 1e-3, 1e-5])
def test_quadratic_relaxation_has_expected_sqrt_epsilon_solution(epsilon_target: float) -> None:
    model, x, y = _quadratic_model()

    result = _solve(
        model,
        epsilon_initial=epsilon_target,
        epsilon_target=epsilon_target,
        seed=31,
    )

    displacement = sqrt(epsilon_target) / 2.0
    np.testing.assert_allclose([x.value, y.value], [displacement, -displacement], atol=_ANALYTIC_ATOL)
    oracle = _check_against_numerical_oracles(model, result)
    assert oracle.source_gap == pytest.approx(epsilon_target, rel=2e-2, abs=3e-6)


def test_quadratic_relaxation_converges_at_sqrt_epsilon_rate() -> None:
    distances: list[float] = []
    for epsilon_target in (1e-1, 1e-3, 1e-5):
        model, x, y = _quadratic_model()
        result = _solve(
            model,
            epsilon_initial=epsilon_target,
            epsilon_target=epsilon_target,
            seed=37,
        )
        assert result.succeeded
        distances.append(float(np.linalg.norm([x.value, y.value])))

    expected = np.sqrt(np.array([1e-1, 1e-3, 1e-5]) / 2.0)
    np.testing.assert_allclose(distances, expected, rtol=2e-2, atol=2e-3)
