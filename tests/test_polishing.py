"""Solver-independent tests for fixed-upper lower-level polishing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from blvpy import BilevelProblem, BilevelResult, LowerProblem, PolishResult, Residuals
from blvpy.errors import SolveError, SolverUnavailableError
from blvpy.polishing import _objective_improvement_ratio


def _scalar_model(
    *,
    maximize: bool = False,
    coefficient: float | None = None,
    offset: float = 0.0,
    upper_floor: float | None = None,
) -> tuple[BilevelProblem, cp.Variable, cp.Variable, cp.Parameter, cp.Parameter]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    fixed_weight = cp.Parameter(nonneg=True, value=1.0, name="fixed_weight")
    lower = LowerProblem(
        cp.Minimize(fixed_weight * y),
        [y >= x],
        parameters=[x],
    )
    upper_expression = cp.square(x) + cp.square(y) if coefficient is None else coefficient * y + offset
    upper_objective = cp.Maximize(upper_expression) if maximize else cp.Minimize(upper_expression)
    upper_constraints = () if upper_floor is None else (y >= upper_floor,)
    model = BilevelProblem(upper_objective, lower, upper_constraints)
    linked_parameter = next(iter(model._parameter_links))
    return model, x, y, fixed_weight, linked_parameter


def _result(
    x: cp.Variable,
    y: cp.Variable,
    *,
    status: str = "optimal",
    x_value: float = 0.4,
    y_value: float = 0.65,
    objective: float | None = None,
    feasibility_tolerance: float = 1e-7,
) -> BilevelResult:
    return BilevelResult(
        status=status,
        objective=objective,
        variable_values={x: np.array(x_value), y: np.array(y_value)},
        _feasibility_tolerance=feasibility_tolerance,
    )


def _snapshot(value: ArrayLike | None) -> NDArray[np.float64] | None:
    return None if value is None else np.array(value, dtype=float, copy=True)


def _seed_non_result_state(
    model: BilevelProblem,
    x: cp.Variable,
    y: cp.Variable,
    fixed_weight: cp.Parameter,
    linked_parameter: cp.Parameter,
) -> dict[cp.Variable | cp.Parameter, NDArray[np.float64] | None]:
    model.validate()
    lifted = model._lifted_problem
    x.value = 1.2
    y.value = 1.3
    fixed_weight.value = 2.0
    linked_parameter.value = 0.9
    lifted.primal.value = np.arange(lifted.primal.size, dtype=float).reshape(lifted.primal.shape)
    lifted.slack.value = np.arange(lifted.slack.size, dtype=float).reshape(lifted.slack.shape) + 10.0
    lifted.dual.value = np.arange(lifted.dual.size, dtype=float).reshape(lifted.dual.shape) + 20.0
    lifted.epsilon.value = 0.03
    leaves = (
        x,
        y,
        fixed_weight,
        linked_parameter,
        lifted.primal,
        lifted.slack,
        lifted.dual,
        lifted.epsilon,
    )
    return {leaf: _snapshot(leaf.value) for leaf in leaves}


def _assert_state(
    expected: dict[cp.Variable | cp.Parameter, NDArray[np.float64] | None],
) -> None:
    for leaf, value in expected.items():
        if value is None:
            assert leaf.value is None
        else:
            np.testing.assert_array_equal(leaf.value, value)


def _sparse_matrix(values: ArrayLike) -> sp.coo_array:
    indices = (np.array([0, 1]), np.array([0, 1]))
    return sp.coo_array((np.asarray(values, dtype=float), indices), shape=(2, 2))


def _sparse_state(
    leaf: cp.Variable | cp.Parameter,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    value = leaf.value_sparse
    assert value is not None
    return (
        np.array(value.row, dtype=np.int64, copy=True),
        np.array(value.col, dtype=np.int64, copy=True),
        np.array(value.data, dtype=float, copy=True),
    )


def _assert_sparse_state(
    leaf: cp.Variable | cp.Parameter,
    expected: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]],
) -> None:
    actual = _sparse_state(leaf)
    for actual_part, expected_part in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_part, expected_part)


def _sparse_model_and_result() -> tuple[
    BilevelProblem,
    cp.Variable,
    cp.Variable,
    cp.Parameter,
    cp.Parameter,
    BilevelResult,
]:
    indices = ([0, 1], [0, 1])
    x = cp.Variable((2, 2), sparsity=indices, name="sparse_x")
    y = cp.Variable(name="sparse_y")
    fixed_shift = cp.Parameter((2, 2), sparsity=indices, name="sparse_fixed_shift")
    fixed_shift.value_sparse = _sparse_matrix([0.4, 0.1])
    lower = LowerProblem(
        cp.Minimize(y),
        [y >= cp.sum(x) + cp.sum(fixed_shift)],
        parameters=[x],
    )
    model = BilevelProblem(cp.Minimize(cp.square(y)), lower)
    model.validate()
    linked_parameter = next(iter(model._parameter_links))
    result = BilevelResult(
        status="optimal",
        variable_values={x: np.diag([0.1, 0.2]), y: np.array(1.05)},
        canonical_primal=(1.05,),
        slack=(0.25,),
        dual=(1.0,),
    )
    return model, x, y, fixed_shift, linked_parameter, result


def test_polish_resolves_lower_at_fixed_upper_and_reevaluates_stale_objective() -> None:
    model, x, y, _, _ = _scalar_model()
    original = _result(x, y, objective=999.0)

    polished = model.polish(original, verbose=False)

    assert isinstance(polished, PolishResult)
    assert polished.feasible
    assert set(polished.variable_values) == {x, y}
    assert float(polished.variable_values[x]) == pytest.approx(0.4, abs=1e-12)
    assert float(polished.variable_values[y]) == pytest.approx(0.4, abs=1e-7)
    assert polished.objective == pytest.approx(0.32, abs=1e-7)
    expected_ratio = (0.4**2 + 0.65**2 - polished.objective) / (0.4**2 + 0.65**2)
    assert polished.objective_improvement_ratio == pytest.approx(expected_ratio, abs=1e-7)
    assert original.objective == 999.0


def test_polish_resolves_maximization_lower_problem() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(
        cp.Maximize(-cp.square(y - x)),
        parameters=[x],
    )
    model = BilevelProblem(cp.Minimize(cp.square(y - 1.0)), lower)
    original = BilevelResult(
        status="optimal",
        variable_values={x: np.array(0.4), y: np.array(-0.2)},
    )

    polished = model.polish(original, verbose=False)

    assert polished.feasible
    assert float(polished.variable_values[x]) == pytest.approx(0.4, abs=1e-12)
    assert float(polished.variable_values[y]) == pytest.approx(0.4, abs=1e-7)
    assert polished.objective == pytest.approx(0.36, abs=1e-7)
    assert polished.objective_improvement_ratio == pytest.approx(0.75, abs=1e-7)


def test_polish_uses_frozen_fixed_parameter_for_solve_and_objective_then_restores_state() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    fixed_shift = cp.Parameter(value=1.5, name="fixed_shift")
    lower = LowerProblem(
        cp.Minimize(cp.square(y - x - fixed_shift)),
        parameters=[x],
    )
    model = BilevelProblem(cp.Minimize(cp.square(y) + fixed_shift), lower)
    model.validate()
    canonical = model.canonicalize()
    linked_parameter = next(iter(model._parameter_links))
    lifted = model._lifted_problem
    assert float(canonical.fixed_parameter_values[fixed_shift.id]) == pytest.approx(1.5)

    fixed_shift.value = -4.0
    linked_parameter.value = 8.0
    leaves = (
        x,
        y,
        fixed_shift,
        linked_parameter,
        lifted.primal,
        lifted.slack,
        lifted.dual,
        lifted.epsilon,
    )
    state = {leaf: _snapshot(leaf.value) for leaf in leaves}
    original = BilevelResult(
        status="optimal",
        variable_values={x: np.array(0.25), y: np.array(0.5)},
    )

    polished = model.polish(original, verbose=False)

    assert polished.feasible
    assert float(polished.variable_values[x]) == pytest.approx(0.25, abs=1e-12)
    assert float(polished.variable_values[y]) == pytest.approx(1.75, abs=1e-7)
    original_objective = 0.5**2 + 1.5
    expected_objective = 1.75**2 + 1.5
    assert polished.objective == pytest.approx(expected_objective, abs=1e-7)
    assert polished.objective_improvement_ratio == pytest.approx(
        (original_objective - expected_objective) / original_objective,
        abs=1e-7,
    )
    _assert_state(state)


@pytest.mark.parametrize(
    ("maximize", "coefficient", "offset", "expected_sign"),
    [
        (False, 1.0, 0.0, 1),
        (False, 1.0, -1.0, 1),
        (False, -1.0, 0.0, -1),
        (True, -1.0, 0.0, 1),
        (True, 1.0, 0.0, -1),
        (False, 0.0, 1.0, 0),
    ],
)
def test_public_ratio_respects_objective_direction_for_positive_and_negative_objectives(
    maximize: bool,
    coefficient: float,
    offset: float,
    expected_sign: int,
) -> None:
    model, x, y, _, _ = _scalar_model(
        maximize=maximize,
        coefficient=coefficient,
        offset=offset,
    )
    original = _result(x, y, objective=None)

    polished = model.polish(original, verbose=False)

    original_objective = coefficient * 0.65 + offset
    numerator = polished.objective - original_objective if maximize else original_objective - polished.objective
    expected = numerator / abs(original_objective)
    assert polished.objective == pytest.approx(coefficient * 0.4 + offset, abs=1e-7)
    assert polished.objective_improvement_ratio == pytest.approx(expected, abs=1e-7)
    if expected_sign == 0:
        assert polished.objective_improvement_ratio == pytest.approx(0.0, abs=1e-12)
    else:
        assert np.sign(polished.objective_improvement_ratio) == expected_sign


@pytest.mark.parametrize(
    ("original", "polished", "maximize", "expected"),
    [
        (10.0, 8.0, False, 0.2),
        (-10.0, -12.0, False, 0.2),
        (10.0, 12.0, True, 0.2),
        (-10.0, -8.0, True, 0.2),
        (10.0, 12.0, False, -0.2),
        (-10.0, -12.0, True, -0.2),
        (10.0, 10.0, False, 0.0),
        (0.0, 1.0, False, None),
        (-0.0, -1.0, True, None),
        (1e-15, 2e-15, True, 1.0),
    ],
)
def test_ratio_definition_including_zero_and_near_zero_baselines(
    original: float,
    polished: float,
    maximize: bool,
    expected: float | None,
) -> None:
    ratio = _objective_improvement_ratio(original, polished, maximize=maximize)

    if expected is None:
        assert ratio is None
    else:
        assert ratio == pytest.approx(expected)


@pytest.mark.parametrize(
    ("original", "polished", "maximize", "expected"),
    [
        (np.finfo(float).max, -np.finfo(float).max, False, 2.0),
        (-np.finfo(float).max, np.finfo(float).max, False, -2.0),
        (np.finfo(float).max, -np.finfo(float).max, True, -2.0),
        (-np.finfo(float).max, np.finfo(float).max, True, 2.0),
    ],
)
def test_ratio_avoids_intermediate_overflow(
    original: float,
    polished: float,
    maximize: bool,
    expected: float,
) -> None:
    assert _objective_improvement_ratio(original, polished, maximize=maximize) == expected


@pytest.mark.parametrize(("maximize", "expected"), [(False, -np.inf), (True, np.inf)])
def test_ratio_retains_signed_infinity_when_true_magnitude_exceeds_float_range(
    maximize: bool,
    expected: float,
) -> None:
    smallest = np.nextafter(0.0, 1.0)

    assert _objective_improvement_ratio(smallest, 1.0, maximize=maximize) == expected


@pytest.mark.parametrize("ratio", [np.inf, -np.inf])
def test_polish_result_accepts_infinite_improvement_ratio(ratio: float) -> None:
    result = PolishResult(
        variable_values={},
        residuals=Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        feasibility_tolerance=1e-7,
        objective=1.0,
        objective_improvement_ratio=ratio,
    )

    assert result.objective_improvement_ratio == ratio


def test_polish_result_rejects_nan_ratio_and_nonfinite_objective() -> None:
    with pytest.raises(ValueError, match="objective_improvement_ratio must not be NaN"):
        PolishResult(
            variable_values={},
            residuals=Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            feasibility_tolerance=1e-7,
            objective=1.0,
            objective_improvement_ratio=np.nan,
        )
    with pytest.raises(ValueError, match="objective must be finite"):
        PolishResult(
            variable_values={},
            residuals=Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            feasibility_tolerance=1e-7,
            objective=np.inf,
            objective_improvement_ratio=0.0,
        )


def test_polish_result_derives_feasibility_at_the_tolerance_boundary() -> None:
    residuals = Residuals(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.1)

    result = PolishResult(
        variable_values={},
        residuals=residuals,
        feasibility_tolerance=0.1,
        objective=1.0,
        objective_improvement_ratio=0.0,
    )

    assert result.residuals is residuals
    assert result.feasibility_tolerance == 0.1
    assert result.feasible


def test_polish_result_validates_residuals_and_feasibility_tolerance() -> None:
    residuals = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = {
        "variable_values": {},
        "residuals": residuals,
        "feasibility_tolerance": 1e-7,
        "objective": 1.0,
        "objective_improvement_ratio": 0.0,
    }

    with pytest.raises(ValueError, match="residuals must be a Residuals instance"):
        PolishResult(**{**values, "residuals": object()})  # type: ignore[arg-type]
    for invalid in (-1.0, np.inf, np.nan, True, np.bool_(True), "invalid"):
        with pytest.raises(ValueError, match="feasibility_tolerance"):
            PolishResult(**{**values, "feasibility_tolerance": invalid})  # type: ignore[arg-type]


def test_polish_result_residual_diagnostics_are_immutable() -> None:
    residuals = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1e-12, 0.0)
    result = PolishResult(
        variable_values={},
        residuals=residuals,
        feasibility_tolerance=1e-7,
        objective=1.0,
        objective_improvement_ratio=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        result.residuals = Residuals(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.feasibility_tolerance = 1.0  # type: ignore[misc]


def test_exact_zero_public_baseline_returns_none() -> None:
    model, x, y, _, _ = _scalar_model(coefficient=1.0, offset=-0.65)

    polished = model.polish(_result(x, y), verbose=False)

    assert polished.objective_improvement_ratio is None


def test_default_tolerance_marks_upper_constraint_violation_infeasible() -> None:
    model, x, y, _, _ = _scalar_model(upper_floor=0.5)

    polished = model.polish(_result(x, y), verbose=False)

    assert float(polished.variable_values[y]) == pytest.approx(0.4, abs=1e-7)
    assert not polished.feasible


@pytest.mark.parametrize(("tolerance", "expected"), [(1e-7, False), (0.1, True)])
def test_polish_uses_originating_result_feasibility_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    tolerance: float,
    expected: bool,
) -> None:
    model, x, y, _, _ = _scalar_model()
    residuals = Residuals(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0)

    def fixed_residuals(current: BilevelProblem, *, epsilon: float) -> Residuals:
        assert current is model
        assert epsilon == 0.0
        return residuals

    monkeypatch.setattr("blvpy.polishing.compute_residuals", fixed_residuals)

    polished = model.polish(
        _result(x, y, feasibility_tolerance=tolerance),
        verbose=False,
    )

    assert polished.residuals is residuals
    assert polished.feasibility_tolerance == tolerance
    assert polished.feasible is expected


def test_complete_scalar_vector_matrix_snapshots_are_immutable_and_adoptable() -> None:
    scalar = cp.Variable(name="scalar")
    vector = cp.Variable(2, name="vector")
    matrix = cp.Variable((2, 2), name="matrix")
    target = cp.vstack([vector, vector + scalar])
    lower = LowerProblem(
        cp.Minimize(cp.sum_squares(matrix - target)),
        parameters=[scalar, vector],
    )
    model = BilevelProblem(
        cp.Minimize(cp.square(scalar) + cp.sum_squares(vector) + cp.sum_squares(matrix)),
        lower,
    )
    result = BilevelResult(
        status="optimal",
        variable_values={
            scalar: np.array(0.5),
            vector: np.array([1.0, 2.0]),
            matrix: np.zeros((2, 2)),
        },
    )
    scalar.value = -4.0
    vector.value = np.array([-3.0, -2.0])
    matrix.value = np.full((2, 2), -1.0)

    polished = model.polish(result, verbose=False)

    assert set(polished.variable_values) == {scalar, vector, matrix}
    assert polished.variable_values[scalar].shape == ()
    assert polished.variable_values[vector].shape == (2,)
    assert polished.variable_values[matrix].shape == (2, 2)
    assert all(not value.flags.writeable for value in polished.variable_values.values())
    with pytest.raises(TypeError):
        polished.variable_values[scalar] = np.array(1.0)  # type: ignore[index]
    with pytest.raises(ValueError):
        polished.variable_values[matrix][0, 0] = 99.0
    assert isinstance(type(polished).feasible, property)
    assert type(polished).feasible.fset is None

    np.testing.assert_array_equal(scalar.value, np.array(-4.0))
    np.testing.assert_array_equal(vector.value, np.array([-3.0, -2.0]))
    np.testing.assert_array_equal(matrix.value, np.full((2, 2), -1.0))

    for variable, value in polished.variable_values.items():
        variable.project_and_assign(value)
    assert float(scalar.value) == pytest.approx(0.5)
    np.testing.assert_allclose(vector.value, [1.0, 2.0])
    np.testing.assert_allclose(matrix.value, [[1.0, 2.0], [1.5, 2.5]], atol=1e-6)
    matrix.value = np.full((2, 2), -100.0)
    assert polished.variable_values[matrix][0, 0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.filterwarnings("ignore:Reading from a sparse CVXPY expression.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Accessing a sparse CVXPY expression.*:RuntimeWarning")
def test_sparse_linked_and_fixed_values_are_restored_after_successful_polish() -> None:
    model, x, y, fixed_shift, linked_parameter, result = _sparse_model_and_result()
    x.value_sparse = _sparse_matrix([1.2, 1.3])
    fixed_shift.value_sparse = _sparse_matrix([2.0, 3.0])
    linked_parameter.value_sparse = _sparse_matrix([4.0, 5.0])
    states = {leaf: _sparse_state(leaf) for leaf in (x, fixed_shift, linked_parameter)}

    polished = model.polish(result, verbose=False)

    assert polished.feasible
    np.testing.assert_allclose(polished.variable_values[x], np.diag([0.1, 0.2]), atol=1e-12)
    assert float(polished.variable_values[y]) == pytest.approx(0.8, abs=1e-7)
    assert all(not value.flags.writeable for value in polished.variable_values.values())
    with pytest.raises(ValueError):
        polished.variable_values[x][0, 0] = 99.0
    for leaf, state in states.items():
        _assert_sparse_state(leaf, state)


@pytest.mark.filterwarnings("ignore:Reading from a sparse CVXPY expression.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Accessing a sparse CVXPY expression.*:RuntimeWarning")
def test_unset_sparse_linked_values_are_restored_after_polish() -> None:
    model, x, _, _, linked_parameter, result = _sparse_model_and_result()
    leaves = (x, linked_parameter)
    for leaf in leaves:
        leaf.save_value(None)

    polished = model.polish(result, verbose=False)

    assert polished.feasible
    for leaf in leaves:
        assert leaf.value_sparse is None


@pytest.mark.filterwarnings("ignore:Reading from a sparse CVXPY expression.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Accessing a sparse CVXPY expression.*:RuntimeWarning")
def test_sparse_linked_and_fixed_values_are_restored_when_polish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, _, fixed_shift, linked_parameter, result = _sparse_model_and_result()
    x.value_sparse = _sparse_matrix([1.2, 1.3])
    fixed_shift.value_sparse = _sparse_matrix([2.0, 3.0])
    linked_parameter.value_sparse = _sparse_matrix([4.0, 5.0])
    states = {leaf: _sparse_state(leaf) for leaf in (x, fixed_shift, linked_parameter)}

    def fail(*args, **kwargs):
        raise cp.SolverError("synthetic sparse polishing failure")

    monkeypatch.setattr("blvpy.polishing.solve_fixed_lower", fail)

    with pytest.raises(SolveError, match="synthetic sparse polishing failure"):
        model.polish(result, verbose=False)

    for leaf, state in states.items():
        _assert_sparse_state(leaf, state)


def test_nonunique_lower_solution_can_lose_optimistic_upper_feasibility() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    lower = LowerProblem(
        cp.Minimize(0.0 * y + x),
        [y >= 0.0, y <= 1.0],
        parameters=[x],
    )
    model = BilevelProblem(cp.Minimize(y), lower, [y >= 0.8])
    result = BilevelResult(
        status="optimal",
        variable_values={x: np.array(0.4), y: np.array(0.9)},
    )

    polished = model.polish(result, verbose=False)

    assert not polished.feasible
    assert float(polished.variable_values[y]) < 0.8


def test_continuation_failed_result_is_polishable() -> None:
    model, x, y, _, _ = _scalar_model()

    polished = model.polish(_result(x, y, status="continuation_failed"), verbose=False)

    assert polished.feasible
    assert float(polished.variable_values[y]) == pytest.approx(0.4, abs=1e-7)


def test_polish_rejects_non_result_and_unsuitable_status() -> None:
    model, x, y, _, _ = _scalar_model()

    with pytest.raises(TypeError, match="BilevelResult"):
        model.polish(object(), verbose=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="successful or continuation_failed"):
        model.polish(_result(x, y, status="infeasible"), verbose=False)


@pytest.mark.parametrize("malformation", ["missing", "wrong_shape", "nonfinite", "nonvariable"])
def test_polish_rejects_malformed_source_snapshots(malformation: str) -> None:
    model, x, y, _, _ = _scalar_model()
    values: dict[Any, ArrayLike] = {x: np.array(0.4), y: np.array(0.65)}
    if malformation == "missing":
        values.pop(y)
    elif malformation == "wrong_shape":
        values[y] = np.array([0.65])
    elif malformation == "nonfinite":
        values[y] = np.array(np.inf)
    else:
        values = {x: np.array(0.4), "y": np.array(0.65)}
    malformed = BilevelResult(status="optimal", variable_values=values)

    with pytest.raises(ValueError, match="variable_values|source variables|finite|shape"):
        model.polish(malformed, verbose=False)


def test_polish_rejects_result_from_another_problem() -> None:
    first, first_x, first_y, _, _ = _scalar_model()
    second, _, _, _, _ = _scalar_model()

    with pytest.raises(ValueError, match="variable_values|source variables"):
        second.polish(_result(first_x, first_y), verbose=False)

    assert first is not second


def test_polish_rejects_tracked_result_from_another_problem_sharing_variables() -> None:
    first, x, y, _, _ = _scalar_model()
    second = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        LowerProblem(cp.Minimize(y), [y >= x], parameters=[x]),
    )
    foreign = BilevelResult(
        status="optimal",
        variable_values={x: np.array(0.4), y: np.array(0.65)},
        _problem_token=first._result_token,
    )

    with pytest.raises(ValueError, match="different BilevelProblem"):
        second.polish(foreign, verbose=False)


def test_polish_forwards_copied_options_and_independent_solver_verbosity(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    model, x, y, _, _ = _scalar_model()
    original = _result(x, y)
    from blvpy import polishing as polishing_module

    real_solve = polishing_module.solve_fixed_lower
    options = {"custom": 17}
    calls: list[tuple[str, dict[str, int], bool]] = []

    def recording_solve(current, solver, received_options, solver_verbose):
        assert current is model
        assert received_options == options
        assert received_options is not options
        calls.append((solver, dict(received_options), solver_verbose))
        return real_solve(current, cp.CLARABEL, {}, False)

    monkeypatch.setattr(polishing_module, "solve_fixed_lower", recording_solve)

    polished = model.polish(
        original,
        solver="CUSTOM",
        solver_options=options,
        verbose=False,
        solver_verbose=True,
    )

    captured = capfd.readouterr()
    assert polished.feasible
    assert calls == [("CUSTOM", options, True)]
    assert options == {"custom": 17}
    assert captured.out == ""
    assert captured.err == ""


def test_solver_unavailable_propagates_unchanged_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, fixed_weight, linked_parameter = _scalar_model()
    state = _seed_non_result_state(model, x, y, fixed_weight, linked_parameter)
    unavailable = SolverUnavailableError("Clarabel unavailable")

    def fail(*args, **kwargs):
        raise unavailable

    monkeypatch.setattr("blvpy.polishing.solve_fixed_lower", fail)

    with pytest.raises(SolverUnavailableError) as raised:
        model.polish(_result(x, y), verbose=False)

    assert raised.value is unavailable
    _assert_state(state)


@pytest.mark.parametrize("status", [cp.OPTIMAL, cp.INFEASIBLE])
def test_incomplete_or_unsuccessful_certificate_raises_solve_error(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    model, x, y, _, _ = _scalar_model()

    def omit_certificate(problem, solver, options, solver_verbose):
        problem._status = status

    monkeypatch.setattr("blvpy.fixed_lower.solve_conic", omit_certificate)

    with pytest.raises(SolveError, match="fixed-upper lower polishing solve failed"):
        model.polish(_result(x, y), verbose=False)


@pytest.mark.parametrize(
    "source_values",
    [
        {},
        pytest.param({-1: np.array(0.4)}, id="foreign-id"),
    ],
)
def test_incomplete_recovered_values_raise_solve_error(
    monkeypatch: pytest.MonkeyPatch,
    source_values: dict[int, NDArray[np.float64]],
) -> None:
    model, x, y, _, _ = _scalar_model()
    from blvpy import polishing as polishing_module

    real_solve = polishing_module.solve_fixed_lower

    def incomplete(current, solver, options, solver_verbose):
        solution = real_solve(current, solver, options, solver_verbose)
        return type(
            "IncompleteSolution",
            (),
            {
                "status": solution.status,
                "primal": solution.primal,
                "slack": solution.slack,
                "dual": solution.dual,
                "source_values": source_values,
            },
        )()

    monkeypatch.setattr(polishing_module, "solve_fixed_lower", incomplete)

    with pytest.raises(SolveError, match="incomplete recovered"):
        model.polish(_result(x, y), verbose=False)


def test_solver_error_is_wrapped_and_complete_state_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, fixed_weight, linked_parameter = _scalar_model()
    state = _seed_non_result_state(model, x, y, fixed_weight, linked_parameter)

    def fail(*args, **kwargs):
        raise cp.SolverError("synthetic numerical failure")

    monkeypatch.setattr("blvpy.polishing.solve_fixed_lower", fail)

    with pytest.raises(SolveError, match="synthetic numerical failure"):
        model.polish(_result(x, y), verbose=False)

    _assert_state(state)


def test_success_and_feasibility_failure_restore_exact_model_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, fixed_weight, linked_parameter = _scalar_model()
    result = _result(x, y)
    state = _seed_non_result_state(model, x, y, fixed_weight, linked_parameter)

    model.polish(result, verbose=False)
    _assert_state(state)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic residual failure")

    monkeypatch.setattr("blvpy.polishing.compute_residuals", fail)
    with pytest.raises(SolveError, match="synthetic residual failure"):
        model.polish(result, verbose=False)
    _assert_state(state)


@pytest.mark.parametrize("ratio_is_none", [False, True])
def test_default_terminal_output_reports_polished_summary(
    capfd: pytest.CaptureFixture[str],
    ratio_is_none: bool,
) -> None:
    offset = -0.65 if ratio_is_none else 0.0
    model, x, y, _, _ = _scalar_model(coefficient=1.0, offset=offset)

    polished = model.polish(_result(x, y))

    captured = capfd.readouterr()
    assert captured.out == ""
    assert "Polishing" in captured.err
    assert "(BLVPY) Result: feasible=true" in captured.err
    assert f"objective={polished.objective:.3e}" in captured.err
    expected_ratio = "n/a" if ratio_is_none else f"{polished.objective_improvement_ratio:.3e}"
    assert f"improvement_ratio={expected_ratio}" in captured.err
    assert (
        f"(BLVPY) Residuals: max_violation={polished.residuals.max_violation:.3e} | "
        f"feasibility_tolerance={polished.feasibility_tolerance:.3e}"
    ) in captured.err


@pytest.mark.parametrize(("name", "value"), [("verbose", 1), ("solver_verbose", "yes")])
def test_polish_verbosity_flags_must_be_boolean(name: str, value: object) -> None:
    model, x, y, _, _ = _scalar_model()

    with pytest.raises(ValueError, match=rf"{name} must be boolean"):
        model.polish(_result(x, y), **{name: value})  # type: ignore[arg-type]


def test_progress_reporting_failure_does_not_change_polished_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, x, y, _, _ = _scalar_model()

    def fail_report(*args, **kwargs):
        raise RuntimeError("broken terminal")

    monkeypatch.setattr("blvpy.progress.ProgressReporter._write", fail_report)

    polished = model.polish(_result(x, y), verbose=True)

    assert polished.feasible
    assert polished.objective == pytest.approx(0.32, abs=1e-7)
