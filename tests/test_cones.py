from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from blvpy.cones import (
    ConeLayout,
    dual_cone_constraints,
    dual_cone_distance,
    primal_cone_constraints,
    primal_cone_distance,
    soc_distance,
)
from blvpy.result import BilevelResult, GapDiagnostics, IterationRecord, Residuals, RunRecord


def test_layout_preserves_canonical_block_order() -> None:
    layout = ConeLayout(zero=2, nonnegative=3, second_order=(3, 4))

    assert layout.size == 12
    assert layout.zero_slice == slice(0, 2)
    assert layout.nonnegative_slice == slice(2, 5)
    assert layout.second_order_slices == (slice(5, 8), slice(8, 12))
    assert [(block.kind, block.slice) for block in layout.blocks] == [
        ("zero", slice(0, 2)),
        ("nonnegative", slice(2, 5)),
        ("second_order", slice(5, 8)),
        ("second_order", slice(8, 12)),
    ]


def test_layout_reads_cvxpy_dimensions_and_rejects_unsupported_cones() -> None:
    dims = SimpleNamespace(zero=1, nonneg=2, soc=[3], exp=0, psd=[], p3d=[], pnd=[])
    assert ConeLayout.from_dims(dims) == ConeLayout(1, 2, (3,))

    dims.exp = 1
    with pytest.raises(ValueError, match="exponential"):
        ConeLayout.from_dims(dims)


def test_primal_and_dual_constraints_use_the_expected_blocks() -> None:
    layout = ConeLayout(zero=1, nonnegative=2, second_order=(3,))
    primal = cp.Variable(layout.size)
    dual = cp.Variable(layout.size)
    primal_constraints = layout.primal_constraints(primal)
    dual_constraints = layout.dual_constraints(dual)

    assert len(primal_constraints) == 3
    assert len(dual_constraints) == 2
    assert primal_constraints[-1].is_dnlp()
    assert dual_constraints[-1].is_dnlp()

    primal.value = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    dual.value = np.array([-100.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in primal_constraints)
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in dual_constraints)


def test_multiple_soc_blocks_accept_boundary_points_in_canonical_order() -> None:
    layout = ConeLayout(zero=2, nonnegative=2, second_order=(3, 4))
    primal = cp.Variable(layout.size)
    dual = cp.Variable(layout.size)
    sqrt_two = np.sqrt(2.0)
    boundary = np.array(
        [
            0.0,
            0.0,
            0.0,
            2.0,
            sqrt_two,
            1.0,
            1.0,
            3.0,
            1.0,
            2.0,
            2.0,
        ]
    )
    primal.value = boundary
    dual.value = np.array([-11.0, 9.0, *boundary[2:]])

    assert layout.primal_distance(boundary) == pytest.approx(0.0, abs=1e-14)
    assert layout.dual_distance(dual.value) == pytest.approx(0.0, abs=1e-14)
    assert all(
        np.max(constraint.violation()) <= 1e-12
        for constraint in (*layout.primal_constraints(primal), *layout.dual_constraints(dual))
    )


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([2.0, 1.0, 1.0], 0.0),
        ([-2.0, 1.0, 1.0], np.sqrt(6.0)),
        ([0.0, 1.0, 0.0], 1.0 / np.sqrt(2.0)),
    ],
)
def test_soc_distance_covers_projection_regions(point: list[float], expected: float) -> None:
    assert soc_distance(point) == pytest.approx(expected)


def test_product_cone_distances_distinguish_zero_cone_dual() -> None:
    layout = ConeLayout(zero=1, nonnegative=2, second_order=(3,))
    value = np.array([4.0, -3.0, 2.0, 0.0, 1.0, 0.0])

    assert layout.primal_distance(value) == pytest.approx(np.sqrt(16.0 + 9.0 + 0.5))
    assert layout.dual_distance(value) == pytest.approx(np.sqrt(9.0 + 0.5))


def test_functional_cone_adapters_match_layout_methods() -> None:
    layout = ConeLayout(zero=1, nonnegative=2, second_order=(3,))
    primal = cp.Variable(layout.size)
    dual = cp.Variable(layout.size)
    primal_value = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    dual_value = np.array([-8.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    primal.value = primal_value
    dual.value = dual_value

    functional_primal = primal_cone_constraints(primal, layout)
    functional_dual = dual_cone_constraints(dual, layout)

    assert len(functional_primal) == len(layout.primal_constraints(primal)) == 3
    assert len(functional_dual) == len(layout.dual_constraints(dual)) == 2
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in functional_primal)
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in functional_dual)
    assert primal_cone_distance(primal_value, layout) == pytest.approx(layout.primal_distance(primal_value))
    assert dual_cone_distance(dual_value, layout) == pytest.approx(layout.dual_distance(dual_value))


def test_symbolic_cone_constraints_report_each_infeasible_block() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,))
    primal = cp.Variable(layout.size)
    dual = cp.Variable(layout.size)
    value = np.array([1.0, -2.0, 0.0, 2.0, 0.0])
    primal.value = value
    dual.value = value

    primal_violations = [float(np.max(constraint.violation())) for constraint in layout.primal_constraints(primal)]
    dual_violations = [float(np.max(constraint.violation())) for constraint in layout.dual_constraints(dual)]

    assert primal_violations == pytest.approx([1.0, 2.0, 2.0])
    assert dual_violations == pytest.approx([2.0, 2.0])


def test_empty_cone_layout_has_no_constraints_or_distance() -> None:
    layout = ConeLayout()
    empty = np.empty(0)

    assert layout.size == 0
    assert layout.blocks == ()
    assert layout.primal_constraints(empty) == ()
    assert layout.dual_constraints(empty) == ()
    assert layout.primal_distance(empty) == 0.0
    assert layout.dual_distance(empty) == 0.0
    assert layout.complementarity(empty, empty) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"zero": -1}, "zero must be a nonnegative integer"),
        ({"nonnegative": 1.5}, "nonnegative must be a nonnegative integer"),
        ({"second_order": (1,)}, "dimension at least 2"),
        ({"second_order": 3}, "second_order must be a sequence"),
    ],
)
def test_layout_rejects_invalid_dimensions(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ConeLayout(**kwargs)


def test_cone_operations_reject_wrong_vector_sizes() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,))

    with pytest.raises(ValueError, match=r"4 entries; expected 5"):
        layout.primal_constraints(np.zeros(4))
    with pytest.raises(ValueError, match=r"6 entries; expected 5"):
        layout.dual_constraints(np.zeros(6))
    with pytest.raises(ValueError, match=r"4 entries; expected 5"):
        layout.primal_distance(np.zeros(4))
    with pytest.raises(ValueError, match=r"6 entries; expected 5"):
        layout.dual_distance(np.zeros(6))
    with pytest.raises(ValueError, match=r"4 entries; expected 5"):
        layout.complementarity(np.zeros(4), np.zeros(5))
    with pytest.raises(ValueError, match="at least two entries"):
        soc_distance([1.0])


def test_cone_operations_reject_complex_vectors() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,))
    symbolic = cp.Variable(layout.size, complex=True)
    numeric = np.ones(layout.size, dtype=complex)

    with pytest.raises(ValueError, match="real-valued"):
        layout.primal_constraints(symbolic)
    with pytest.raises(ValueError, match="real-valued"):
        layout.dual_constraints(symbolic)
    with pytest.raises(ValueError, match="real-valued"):
        layout.primal_distance(numeric)
    with pytest.raises(ValueError, match="real-valued"):
        layout.dual_distance(numeric)
    with pytest.raises(ValueError, match="real-valued"):
        layout.complementarity(numeric, np.ones(layout.size))
    with pytest.raises(ValueError, match="real-valued"):
        soc_distance(np.array([1.0, 1.0j]))


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_constrained_blocks_have_infinite_distance(nonfinite: float) -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,))
    nonnegative_nonfinite = np.array([0.0, nonfinite, 2.0, 0.0, 0.0])
    soc_nonfinite = np.array([0.0, 0.0, 2.0, nonfinite, 0.0])

    assert np.isinf(layout.primal_distance(nonnegative_nonfinite))
    assert np.isinf(layout.dual_distance(nonnegative_nonfinite))
    assert np.isinf(layout.primal_distance(soc_nonfinite))
    assert np.isinf(layout.dual_distance(soc_nonfinite))
    assert np.isinf(soc_distance([2.0, nonfinite]))


def _independent_product_cone_distance(point: np.ndarray, *, dual: bool) -> float:
    projected = cp.Variable(12)
    constraints: list[cp.Constraint] = [
        projected[2:5] >= 0,
        cp.SOC(projected[5], projected[6:8]),
        cp.SOC(projected[8], projected[9:12]),
    ]
    if not dual:
        constraints.insert(0, projected[:2] == 0)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(projected - point)), constraints)

    problem.solve(solver=cp.CLARABEL)

    assert problem.status in cp.settings.SOLUTION_PRESENT
    return float(np.sqrt(max(float(problem.value), 0.0)))


def test_product_cone_distances_match_independent_cvxpy_projections() -> None:
    layout = ConeLayout(zero=2, nonnegative=3, second_order=(3, 4))
    points = np.random.default_rng(90210).normal(size=(6, layout.size))

    for point in points:
        assert layout.primal_distance(point) == pytest.approx(
            _independent_product_cone_distance(point, dual=False),
            abs=2e-6,
        )
        assert layout.dual_distance(point) == pytest.approx(
            _independent_product_cone_distance(point, dual=True),
            abs=2e-6,
        )


def test_complementarity_uses_unmodified_canonical_order() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,))
    primal = np.array([0.0, 2.0, 3.0, 1.0, 1.0])
    dual = np.array([7.0, 5.0, 4.0, -1.0, 2.0])

    assert layout.complementarity(primal, dual) == pytest.approx(primal @ dual)


def test_gap_diagnostics_verify_inexact_identity() -> None:
    a = np.array([[1.5, -0.5], [0.25, 2.0], [-1.0, 1.0]])
    b = np.array([1.0, -2.0, 0.5])
    c = np.array([0.75, -1.25])
    primal = np.array([0.4, -0.7])
    slack = np.array([0.2, 0.3, -0.1])
    dual = np.array([0.8, -0.2, 1.1])
    primal_residual = a @ primal + slack - b
    dual_residual = a.T @ dual + c

    diagnostics = GapDiagnostics.from_canonical(
        c=c,
        b=b,
        primal=primal,
        dual=dual,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        complementarity=slack @ dual,
    )

    assert diagnostics.identity_error == pytest.approx(0.0, abs=1e-14)


def test_result_snapshots_arrays_and_exposes_history() -> None:
    residuals = Residuals(1e-8, 2e-8, 0.0, 0.0, 0.0, 0.0, 5e-7, 0.0)
    record = IterationRecord(1e-6, "optimal", 2.0, residuals)
    source = np.array([1.0, 2.0])
    initial = np.array([-1.0, 1.0])
    failed_record = IterationRecord(1e-2, "solver_error", None, residuals)
    failed_run = RunRecord(
        index=0,
        initial_values={"x": np.zeros(2)},
        status="continuation_failed",
        objective=None,
        iterations=(failed_record,),
    )
    selected_run = RunRecord(
        index=1,
        initial_values={"x": initial},
        status="optimal",
        objective=2.0,
        iterations=(record,),
    )
    result = BilevelResult(
        status="optimal",
        objective=2.0,
        variable_values={"x": source},
        canonical_primal=source,
        slack=np.ones(2),
        dual=np.ones(2),
        iterations=(record,),
        runs=(failed_run, selected_run),
        selected_run_index=1,
    )
    source[0] = 99.0
    initial[0] = 99.0

    assert result.epsilon_history == (1e-6,)
    assert result.complementarity == 5e-7
    assert result.succeeded
    assert not result.certified
    assert result.selected_run is selected_run
    assert result.all_objectives == (None, 2.0)
    assert selected_run.epsilon_history == (1e-6,)
    assert selected_run.attempted_epsilon_history == (1e-6,)
    assert selected_run.solver_statuses == ("optimal",)
    assert selected_run.final_iteration is record
    assert selected_run.residuals is residuals
    assert selected_run.complementarity == 5e-7
    assert selected_run.final_epsilon == 1e-6
    assert selected_run.succeeded
    assert selected_run.initial_values["x"].tolist() == [-1.0, 1.0]
    assert result.variable_values["x"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        result.canonical_primal[0] = 10.0
    with pytest.raises(ValueError):
        selected_run.initial_values["x"][0] = 10.0


def test_run_history_tracks_accepted_and_attempted_epsilons() -> None:
    residuals = Residuals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    iterations = (
        IterationRecord(1e-1, "optimal", 3.0, residuals),
        IterationRecord(1e-2, "solver_error", None, residuals),
        IterationRecord(3e-2, "optimal_inaccurate", 2.5, residuals),
        IterationRecord(1e-2, "optimal", 2.0, residuals),
    )
    run = RunRecord(
        index=np.int64(2),
        initial_values={"x": 0.0},
        status="optimal",
        objective=2.0,
        iterations=iterations,
    )

    assert run.index == 2
    assert run.epsilon_history == (1e-1, 3e-2, 1e-2)
    assert run.attempted_epsilon_history == (1e-1, 1e-2, 3e-2, 1e-2)
    assert run.final_iteration is iterations[-1]


def test_result_validates_run_selection() -> None:
    run = RunRecord(index=3, initial_values={}, status="failed")

    with pytest.raises(ValueError, match="recorded runs"):
        BilevelResult(status="failed", runs=(run,), selected_run_index=0)
    with pytest.raises(ValueError, match="unique"):
        BilevelResult(status="failed", runs=(run, run))
