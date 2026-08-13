from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from blvpy.cones import ConeLayout, soc_distance
from blvpy.result import BilevelResult, GapDiagnostics, IterationRecord, Residuals


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
    result = BilevelResult(
        status="optimal",
        objective=2.0,
        variable_values={"x": source},
        canonical_primal=source,
        slack=np.ones(2),
        dual=np.ones(2),
        iterations=(record,),
    )
    source[0] = 99.0

    assert result.epsilon_history == (1e-6,)
    assert result.complementarity == 5e-7
    assert result.succeeded
    assert not result.certified
    assert result.variable_values["x"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        result.canonical_primal[0] = 10.0
