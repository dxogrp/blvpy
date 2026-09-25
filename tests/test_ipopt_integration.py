"""Native IPOPT end-to-end checks against independent numerical oracles.

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

import blvpy.continuation as continuation
from blvpy import BilevelProblem, GapDiagnostics, LowerProblem, Residuals

if TYPE_CHECKING:
    from blvpy import BilevelResult, ConeLayout

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


def _exp_distance(point: ArrayLike, *, dual: bool) -> float:
    """Independent Euclidean distance oracle using CVXPY's native cone."""

    vector = np.asarray(point, dtype=float).reshape(3)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(vector))))
    if dual:
        u, v, w = (float(entry) for entry in vector)
        if u <= tolerance and w >= -tolerance:
            if abs(u) <= tolerance and v >= -tolerance:
                return 0.0
            if u < 0.0 and w > 0.0 and (-u) * np.log((-u) / w) <= v - u + tolerance:
                return 0.0
    else:
        x, y, z = (float(entry) for entry in vector)
        if y >= -tolerance and z >= -tolerance:
            if abs(y) <= tolerance and x <= tolerance:
                return 0.0
            if y > 0.0 and z > 0.0 and x <= y * (np.log(z) - np.log(y)) + tolerance:
                return 0.0
    projected = cp.Variable(3)
    if dual:
        cone = cp.ExpCone(-projected[1], -projected[0], np.e * projected[2])
    else:
        cone = cp.ExpCone(projected[0], projected[1], projected[2])
    projection = cp.Problem(cp.Minimize(cp.sum_squares(projected - vector)), [cone])
    projection.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert projection.status in cp.settings.SOLUTION_PRESENT
    assert projection.value is not None
    return sqrt(max(float(projection.value), 0.0))


def _power_3d_distance(point: ArrayLike, alpha: float, *, dual: bool) -> float:
    """Independent Euclidean distance oracle using CVXPY's native cone."""

    vector = np.asarray(point, dtype=float).reshape(3)
    x_head = float(vector[0] / alpha) if dual else float(vector[0])
    y_head = float(vector[1] / (1.0 - alpha)) if dual else float(vector[1])
    tail = abs(float(vector[2]))
    if x_head >= 0.0 and y_head >= 0.0:
        if tail == 0.0:
            return 0.0
        if x_head > 0.0 and y_head > 0.0:
            log_bound = alpha * np.log(x_head) + (1.0 - alpha) * np.log(y_head)
            if np.log(tail) <= log_bound + 1e-10:
                return 0.0
    projected = cp.Variable(3)
    if dual:
        cone = cp.PowCone3D(projected[0] / alpha, projected[1] / (1.0 - alpha), projected[2], alpha)
    else:
        cone = cp.PowCone3D(projected[0], projected[1], projected[2], alpha)
    projection = cp.Problem(cp.Minimize(cp.sum_squares(projected - vector)), [cone])
    projection.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert projection.status in cp.settings.SOLUTION_PRESENT
    assert projection.value is not None
    return sqrt(max(float(projection.value), 0.0))


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
    for block in layout.exponential_slices:
        squared += _exp_distance(vector[block], dual=dual) ** 2
    for block, alpha in zip(layout.power_3d_slices, layout.power_3d, strict=True):
        squared += _power_3d_distance(vector[block], alpha, dual=dual) ** 2
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
    constraints.extend(
        cp.ExpCone(value[block.start], value[block.start + 1], value[block.start + 2])
        for block in layout.exponential_slices
    )
    constraints.extend(
        cp.PowCone3D(value[block.start], value[block.start + 1], value[block.start + 2], alpha)
        for block, alpha in zip(layout.power_3d_slices, layout.power_3d, strict=True)
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
    normalized_direct_value = (
        -direct_lower_value if isinstance(model.lower_problem.objective, cp.Maximize) else direct_lower_value
    )
    assert canonical_reference.value == pytest.approx(normalized_direct_value, abs=1e-8)

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


def _assert_gap_diagnostics_match(
    actual: GapDiagnostics,
    expected: GapDiagnostics,
    *,
    source_gap_atol: float = 2e-8,
) -> None:
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
    assert actual.source_gap == pytest.approx(expected.source_gap, abs=source_gap_atol)


def _check_against_numerical_oracles(
    model: BilevelProblem,
    result: BilevelResult,
    *,
    check_reference_recovery: bool = True,
    check_gap_convenience: bool = False,
    canonical_source_atol: float = 1e-8,
    gap_convenience_atol: float = 2e-8,
) -> _NumericalOracle:
    assert result.succeeded
    assert result.objective is not None
    assert result.residuals is not None
    assert result.final_epsilon is not None

    source_values = _snapshot_source_values(model, result)
    _assign_source_values(source_values)
    evaluated_upper_objective = float(model.upper_objective.value)
    assert result.objective == pytest.approx(evaluated_upper_objective, abs=1e-9)
    returned_lower_value = float(model.lower_problem.objective.expr.value)

    parameter_values = {parameter: source_values[variable] for parameter, variable in model._parameter_links.items()}
    data = model.canonicalize().apply_numeric(parameter_values)
    canonical_primal = np.asarray(result.canonical_primal, dtype=float).reshape(-1)
    returned_canonical_value = float(data.c @ canonical_primal + data.d)
    normalized_returned_value = (
        -returned_lower_value if isinstance(model.lower_problem.objective, cp.Maximize) else returned_lower_value
    )
    assert returned_canonical_value == pytest.approx(normalized_returned_value, abs=canonical_source_atol)

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

    source_gap = (
        direct_lower_value - returned_lower_value
        if isinstance(model.lower_problem.objective, cp.Maximize)
        else returned_lower_value - direct_lower_value
    )
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
        _assert_gap_diagnostics_match(
            model.gap_diagnostics(result),
            gap_diagnostics,
            source_gap_atol=gap_convenience_atol,
        )
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
    feasibility_tolerance: float = 1e-7,
    seed: int = 0,
) -> BilevelResult:
    return model.solve(
        epsilon_initial=epsilon_initial,
        epsilon_target=epsilon_target,
        best_of=best_of,
        feasibility_tolerance=feasibility_tolerance,
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


def _vectorized_exponential_cone_model() -> tuple[
    BilevelProblem,
    cp.Variable,
    cp.Variable,
    cp.Variable,
    cp.Variable,
    NDArray[np.float64],
]:
    base = np.array([0.5, 1.0, 2.0])
    x = cp.Variable(name="x", bounds=[-0.5, 0.5])
    exponent = cp.Variable(3, name="exponent")
    scale = cp.Variable(3, name="scale")
    epigraph = cp.Variable(3, name="epigraph")
    lower = LowerProblem(
        cp.Minimize(cp.sum(epigraph)),
        [
            exponent == x + np.log(base),
            scale == 1.0,
            cp.ExpCone(exponent, scale, epigraph),
        ],
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(
            cp.square(x)
            + cp.sum_squares(exponent - np.log(base))
            + cp.sum_squares(scale - 1.0)
            + cp.sum_squares(epigraph - base)
        ),
        lower,
    )
    return model, x, exponent, scale, epigraph, base


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


def test_quadratic_solve_then_polish_preserves_state_and_uses_solve_tolerance() -> None:
    model, x, y = _quadratic_model()
    epsilon = 1e-4
    feasibility_tolerance = 5e-6

    result = _solve(
        model,
        epsilon_initial=epsilon,
        epsilon_target=epsilon,
        feasibility_tolerance=feasibility_tolerance,
        seed=41,
    )

    assert result.succeeded
    assert result._feasibility_tolerance == feasibility_tolerance
    lifted = model._lifted_problem
    leaves = (
        *model.source_variables,
        *model._cvxpy_lower_problem.parameters(),
        lifted.primal,
        lifted.slack,
        lifted.dual,
        lifted.epsilon,
    )
    model_state = {leaf: None if leaf.value is None else np.array(leaf.value, copy=True) for leaf in leaves}
    result_values = {variable: np.array(value, copy=True) for variable, value in result.variable_values.items()}
    result_vectors = {
        name: np.array(getattr(result, name), copy=True) for name in ("canonical_primal", "slack", "dual")
    }
    original_x = float(result.variable_values[x])
    original_y = float(result.variable_values[y])
    original_objective = (original_x - 1.0) ** 2 + (original_y + 1.0) ** 2

    polished = model.polish(result, solver=cp.CLARABEL, verbose=False)

    assert polished.feasible
    assert set(polished.variable_values) == {x, y}
    assert float(polished.variable_values[x]) == pytest.approx(original_x, abs=1e-12)
    assert float(polished.variable_values[y]) == pytest.approx(original_x, abs=1e-7)
    assert abs(float(polished.variable_values[y]) - original_y) > 0.5 * sqrt(epsilon)
    polished_objective = (original_x - 1.0) ** 2 + (original_x + 1.0) ** 2
    assert result.objective == pytest.approx(original_objective, abs=1e-9)
    assert polished.objective == pytest.approx(polished_objective, abs=1e-8)
    expected_ratio = (original_objective - polished_objective) / abs(original_objective)
    assert polished.objective_improvement_ratio == pytest.approx(expected_ratio, abs=1e-8)
    assert polished.objective_improvement_ratio < 0.0

    for leaf, value in model_state.items():
        if value is None:
            assert leaf.value is None
        else:
            np.testing.assert_array_equal(leaf.value, value)
    assert set(result.variable_values) == set(result_values)
    for variable, value in result_values.items():
        np.testing.assert_array_equal(result.variable_values[variable], value)
    for name, value in result_vectors.items():
        np.testing.assert_array_equal(getattr(result, name), value)
    assert result.objective == pytest.approx(original_objective, abs=1e-9)
    assert result._feasibility_tolerance == feasibility_tolerance


def test_analytic_max_max_preserves_original_objective_values() -> None:
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    lower_objective = cp.Maximize(-cp.square(y - x))
    lower = LowerProblem(lower_objective, parameters=[x])
    upper_objective = cp.Maximize(-cp.square(x - 1.0) - cp.square(y + 1.0))
    model = BilevelProblem(upper_objective, lower)

    result = _solve(model, seed=5)

    assert result.final_epsilon == pytest.approx(1e-5)
    displacement = sqrt(result.final_epsilon) / 2.0
    np.testing.assert_allclose([x.value, y.value], [displacement, -displacement], atol=1e-3)
    expected_upper_value = -2.0 + 2.0 * sqrt(result.final_epsilon) - result.final_epsilon / 2.0
    expected_lower_value = -result.final_epsilon
    assert result.objective == pytest.approx(expected_upper_value, abs=2e-3)
    assert result.selected_run is not None
    assert result.selected_run.objective == pytest.approx(result.objective, abs=1e-9)
    assert result.final_iteration is not None
    assert result.final_iteration.objective == pytest.approx(result.objective, abs=1e-9)
    assert lower_objective.value == pytest.approx(expected_lower_value, abs=3e-6)

    oracle = _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
    )
    assert oracle.direct_lower_value == pytest.approx(0.0, abs=1e-8)
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
    upper_constraint = x <= 1.0
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 2.0) + cp.sum_squares(y - np.array([1.0, 0.0])) + cp.square(t - 1.0)),
        lower,
        upper_constraints=[upper_constraint],
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
    assert float(np.asarray(upper_constraint.violation())) <= 1e-7
    assert abs(float(x.value) - 1.0) <= _ANALYTIC_ATOL
    assert abs(float(t.value) - np.linalg.norm(y.value)) <= _ANALYTIC_ATOL
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
    )


def test_exact_geometric_mean_and_rational_power_lower_problem() -> None:
    x = cp.Variable(name="x", bounds=[0.5, 1.5])
    y = cp.Variable(2, nonneg=True, name="y")
    z = cp.Variable(name="z")
    lower = LowerProblem(
        cp.Minimize(-cp.geo_mean(y) + cp.power(z, 1.5)),
        [cp.sum(y) == 2.0 * x, z >= x],
        parameters=[x],
    )
    target = 0.9
    model = BilevelProblem(
        cp.Minimize(cp.square(x - target) + cp.sum_squares(y - target) + cp.square(z - target)),
        lower,
    )

    canonical = model.canonicalize()
    assert len(canonical.cone_layout.second_order) >= 3
    result = _solve(
        model,
        epsilon_initial=1e-5,
        epsilon_target=1e-5,
        seed=9,
    )

    np.testing.assert_allclose([x.value, *y.value, z.value], np.full(4, target), atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
    )


def test_exact_exp_lower_problem_uses_multiple_exponential_cones_end_to_end() -> None:
    projection_tolerance = 5e-5
    offsets = np.array([-0.6, 0.0, 0.5])
    target = 0.2
    x = cp.Variable(name="x", bounds=[-0.5, 1.0])
    y = cp.Variable(3, name="y")
    lower = LowerProblem(
        cp.Minimize(cp.sum(cp.exp(y))),
        [y >= x + offsets],
        parameters=[x],
    )
    expected_y = target + offsets
    model = BilevelProblem(
        cp.Minimize(cp.square(x - target) + cp.sum_squares(y - expected_y)),
        lower,
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.exponential == 3
    assert canonical.cone_layout.second_order == ()
    assert canonical.cone_layout.power_3d == ()
    result = _solve(
        model,
        epsilon_initial=1e-5,
        epsilon_target=1e-5,
        feasibility_tolerance=projection_tolerance,
        seed=61,
    )

    assert float(x.value) == pytest.approx(target, abs=_ANALYTIC_ATOL)
    np.testing.assert_allclose(y.value, expected_y, atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
        gap_convenience_atol=1e-7,
    )
    polished = model.polish(result, solver=cp.CLARABEL, verbose=False)
    assert polished.feasible
    assert (
        max(
            polished.residuals.primal_equality,
            polished.residuals.dual_equality,
            polished.residuals.recovery,
            polished.residuals.upper_constraints,
            polished.residuals.gap_violation,
        )
        <= 1e-7
    )
    assert polished.residuals.primal_cone <= projection_tolerance
    assert polished.residuals.dual_cone <= projection_tolerance
    np.testing.assert_allclose(polished.variable_values[y], float(polished.variable_values[x]) + offsets, atol=1e-7)


def test_vectorized_direct_exponential_cone_lower_constraint_end_to_end() -> None:
    model, x, exponent, scale, epigraph, base = _vectorized_exponential_cone_model()

    canonical = model.canonicalize()
    assert canonical.cone_layout.exponential == 3
    assert len(canonical.cone_layout.exponential_slices) == 3
    assert all(block.stop - block.start == 3 for block in canonical.cone_layout.exponential_slices)
    result = _solve(model, epsilon_initial=1e-5, epsilon_target=1e-5, seed=67)

    assert float(x.value) == pytest.approx(0.0, abs=_ANALYTIC_ATOL)
    np.testing.assert_allclose(exponent.value, np.log(base), atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(scale.value, np.ones(3), atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(epigraph.value, base, atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
        gap_convenience_atol=1e-7,
    )


def test_exponential_cone_infeasible_start_uses_real_ipopt_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, exponent, scale, epigraph, base = _vectorized_exponential_cone_model()
    original_initialize_lower = continuation._initialize_lower
    original_restore_feasibility = continuation._restore_feasibility
    restorations: list[tuple[Residuals, Residuals]] = []

    def initialize_with_infeasible_exponential_blocks(current, *args, **kwargs) -> None:
        original_initialize_lower(current, *args, **kwargs)
        layout = current.canonicalize().cone_layout
        assert layout.exponential == 3
        lifted = current._lifted_problem
        slack = np.asarray(lifted.slack.value, dtype=float).reshape(-1).copy()
        dual = np.asarray(lifted.dual.value, dtype=float).reshape(-1).copy()
        for block in layout.exponential_slices:
            slack[block] = (0.75, 0.0, 0.0)
            dual[block] = (0.75, 0.0, 0.0)
        lifted.slack.save_value(slack)
        lifted.dual.save_value(dual)

    def record_restoration(
        current,
        epsilon,
        solver,
        options,
        solver_verbose,
        tolerance=1e-7,
    ) -> None:
        assert solver == cp.IPOPT
        before = continuation.compute_residuals(current, epsilon)
        original_restore_feasibility(current, epsilon, solver, options, solver_verbose, tolerance)
        after = continuation.compute_residuals(current, epsilon)
        restorations.append((before, after))

    monkeypatch.setattr(continuation, "_initialize_lower", initialize_with_infeasible_exponential_blocks)
    monkeypatch.setattr(continuation, "_restore_feasibility", record_restoration)

    result = _solve(model, epsilon_initial=1e-5, epsilon_target=1e-5, seed=71)

    assert len(restorations) == 1
    before, after = restorations[0]
    assert before.primal_cone > 0.5
    assert before.dual_cone > 0.5
    assert before.gap_violation > 0.5
    assert after.max_violation <= 1e-7
    assert result.residuals is not None
    assert result.residuals.max_violation <= _FINAL_RESIDUAL_TOL
    assert float(x.value) == pytest.approx(0.0, abs=_ANALYTIC_ATOL)
    np.testing.assert_allclose(exponent.value, np.log(base), atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(scale.value, np.ones(3), atol=_ANALYTIC_ATOL)
    np.testing.assert_allclose(epigraph.value, base, atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
        gap_convenience_atol=1e-7,
    )


def test_exact_power_lower_problem_uses_power_cone_end_to_end() -> None:
    projection_tolerance = 5e-5
    x = cp.Variable(name="x", bounds=[0.25, 1.5])
    y = cp.Variable(nonneg=True, name="y")
    lower = LowerProblem(
        cp.Minimize(cp.power(y, np.sqrt(2.0), approx=False)),
        [y >= x],
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.square(y - 1.0)),
        lower,
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.power_3d
    result = _solve(
        model,
        epsilon_initial=1e-5,
        epsilon_target=1e-5,
        feasibility_tolerance=projection_tolerance,
        seed=43,
    )

    np.testing.assert_allclose([x.value, y.value], [1.0, 1.0], atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
    )
    polished = model.polish(result, solver=cp.CLARABEL, verbose=False)
    assert polished.feasible
    assert (
        max(
            polished.residuals.primal_equality,
            polished.residuals.dual_equality,
            polished.residuals.recovery,
            polished.residuals.upper_constraints,
            polished.residuals.gap_violation,
        )
        <= 1e-7
    )
    assert polished.residuals.primal_cone <= projection_tolerance
    assert polished.residuals.dual_cone <= projection_tolerance
    assert float(polished.variable_values[y]) == pytest.approx(float(polished.variable_values[x]), abs=1e-7)


def test_exact_pnorm_lower_problem_uses_power_cones_end_to_end() -> None:
    exponent = np.sqrt(2.0)
    alpha = 1.0 / exponent
    x = cp.Variable(name="x", bounds=[0.25, 1.5])
    y = cp.Variable(3, nonneg=True, name="y")
    expected_y = np.array([1.0, 0.3, 0.4])
    lower = LowerProblem(
        cp.Minimize(cp.pnorm(y, exponent, approx=False)),
        [y >= cp.hstack([x, 0.3, 0.4])],
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.sum_squares(y - expected_y)),
        lower,
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.second_order == ()
    np.testing.assert_allclose(canonical.cone_layout.power_3d, np.full(3, alpha))
    result = _solve(model, epsilon_initial=1e-5, epsilon_target=1e-5, seed=53)

    actual = np.concatenate(([float(x.value)], np.asarray(y.value, dtype=float)))
    expected = np.concatenate(([1.0], expected_y))
    np.testing.assert_allclose(actual, expected, atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
    )


def test_direct_power_cone_lower_constraint_end_to_end() -> None:
    x = cp.Variable(name="x", bounds=[0.25, 1.5])
    heads = cp.Variable(2, name="heads")
    tail = cp.Variable(name="tail")
    alpha = 0.25
    lower = LowerProblem(
        cp.Maximize(tail),
        [heads[0] == x, heads[1] == 1.0, cp.PowCone3D(heads[0], heads[1], tail, alpha)],
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.sum_squares(heads - 1.0) + cp.square(tail - 1.0)),
        lower,
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.power_3d == (alpha,)
    result = _solve(model, epsilon_initial=1e-5, epsilon_target=1e-5, seed=47)

    assert float(x.value) == pytest.approx(1.0, abs=_ANALYTIC_ATOL)
    np.testing.assert_allclose(heads.value, np.ones(2), atol=_ANALYTIC_ATOL)
    assert float(tail.value) == pytest.approx(1.0, abs=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
    )


def test_power_cone_infeasible_start_uses_real_ipopt_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = cp.Variable(name="x", bounds=[0.25, 1.5])
    heads = cp.Variable(2, name="heads")
    tail = cp.Variable(name="tail")
    alpha = 0.25
    lower = LowerProblem(
        cp.Maximize(tail),
        [heads[0] == x, heads[1] == 1.0, cp.PowCone3D(heads[0], heads[1], tail, alpha)],
        parameters=[x],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(x - 1.0) + cp.sum_squares(heads - 1.0) + cp.square(tail - 1.0)),
        lower,
    )
    original_initialize_lower = continuation._initialize_lower
    original_restore_feasibility = continuation._restore_feasibility
    restorations: list[tuple[Residuals, Residuals]] = []

    def initialize_with_infeasible_power_blocks(current, *args, **kwargs) -> None:
        original_initialize_lower(current, *args, **kwargs)
        layout = current.canonicalize().cone_layout
        assert layout.power_3d == (alpha,)
        lifted = current._lifted_problem
        slack = np.asarray(lifted.slack.value, dtype=float).reshape(-1).copy()
        dual = np.asarray(lifted.dual.value, dtype=float).reshape(-1).copy()
        for block in layout.power_3d_slices:
            slack[block] = (0.0, 0.0, 1.0)
            dual[block] = (0.0, 0.0, 1.0)
        lifted.slack.save_value(slack)
        lifted.dual.save_value(dual)

    def record_restoration(
        current,
        epsilon,
        solver,
        options,
        solver_verbose,
        tolerance=1e-7,
    ) -> None:
        assert solver == cp.IPOPT
        before = continuation.compute_residuals(current, epsilon)
        original_restore_feasibility(current, epsilon, solver, options, solver_verbose, tolerance)
        after = continuation.compute_residuals(current, epsilon)
        restorations.append((before, after))

    monkeypatch.setattr(continuation, "_initialize_lower", initialize_with_infeasible_power_blocks)
    monkeypatch.setattr(continuation, "_restore_feasibility", record_restoration)

    result = _solve(model, epsilon_initial=1e-5, epsilon_target=1e-5, seed=59)

    assert len(restorations) == 1
    before, after = restorations[0]
    assert before.primal_cone > 0.5
    assert before.dual_cone > 0.5
    assert before.gap_violation > 0.5
    assert after.max_violation <= 1e-7
    assert result.residuals is not None
    assert result.residuals.max_violation <= _FINAL_RESIDUAL_TOL
    np.testing.assert_allclose([x.value, *heads.value, tail.value], np.ones(4), atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
    )


def test_cummax_and_dotsort_lower_problem_with_affine_response() -> None:
    x = cp.Variable(name="x", bounds=[-1.0, 1.0])
    y = cp.Variable(3, name="y")
    response = cp.hstack([x, 1.0 - x, 0.5 + 0.25 * x])
    lower = LowerProblem(
        cp.Minimize(cp.sum(cp.cummax(y, axis=0)) + cp.dotsort(y, np.array([-0.5, 0.25, 1.0]))),
        [y == response],
        parameters=[x],
    )
    target = 0.2
    expected_y = np.array([target, 1.0 - target, 0.5 + 0.25 * target])
    model = BilevelProblem(
        cp.Minimize(cp.square(x - target) + cp.sum_squares(y - expected_y)),
        lower,
    )

    canonical = model.canonicalize()
    assert canonical.cone_layout.nonnegative > 0
    assert canonical.cone_layout.second_order == ()
    result = _solve(
        model,
        epsilon_initial=1e-5,
        epsilon_target=1e-5,
        seed=13,
    )

    assert float(x.value) == pytest.approx(target, abs=_ANALYTIC_ATOL)
    np.testing.assert_allclose(y.value, expected_y, atol=_ANALYTIC_ATOL)
    assert result.objective == pytest.approx(0.0, abs=_OBJECTIVE_ATOL)
    _check_against_numerical_oracles(
        model,
        result,
        check_gap_convenience=True,
        canonical_source_atol=2e-5,
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
