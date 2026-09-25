import math
import warnings
from fractions import Fraction
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from blvpy._cones import projection as cone_projection
from blvpy.cones import (
    ConeLayout,
    dual_cone_constraints,
    dual_cone_distance,
    primal_cone_constraints,
    primal_cone_distance,
    soc_distance,
)
from blvpy.result import BilevelResult, IterationRecord, Residuals, RunRecord


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


def test_layout_appends_power_cones_and_exposes_cvxpy_aliases() -> None:
    layout = ConeLayout(2, 3, (3, 4), (0.2, 0.75))

    assert layout.size == 18
    assert layout.p3d == layout.power_3d == (0.2, 0.75)
    assert layout.power_3d_slices == layout.p3d_slices == (slice(12, 15), slice(15, 18))
    assert [(block.kind, block.index, block.slice) for block in layout.blocks] == [
        ("zero", 0, slice(0, 2)),
        ("nonnegative", 0, slice(2, 5)),
        ("second_order", 0, slice(5, 8)),
        ("second_order", 1, slice(8, 12)),
        ("power_3d", 0, slice(12, 15)),
        ("power_3d", 1, slice(15, 18)),
    ]


def test_layout_places_exponential_cones_before_power_cones() -> None:
    layout = ConeLayout(2, 3, (3, 4), (0.2, 0.75), 2)

    assert layout.size == 24
    assert layout.exp == layout.exponential == 2
    assert layout.exponential_slices == layout.exp_slices == (slice(12, 15), slice(15, 18))
    assert layout.power_3d_slices == (slice(18, 21), slice(21, 24))
    assert [(block.kind, block.index, block.slice) for block in layout.blocks] == [
        ("zero", 0, slice(0, 2)),
        ("nonnegative", 0, slice(2, 5)),
        ("second_order", 0, slice(5, 8)),
        ("second_order", 1, slice(8, 12)),
        ("exponential", 0, slice(12, 15)),
        ("exponential", 1, slice(15, 18)),
        ("power_3d", 0, slice(18, 21)),
        ("power_3d", 1, slice(21, 24)),
    ]


def test_layout_reads_cvxpy_dimensions_and_rejects_unsupported_cones() -> None:
    dims = SimpleNamespace(zero=1, nonneg=2, soc=[3], exp=0, psd=[], p3d=[0.25, 0.8], pnd=[])
    assert ConeLayout.from_dims(dims) == ConeLayout(1, 2, (3,), (0.25, 0.8))

    for alias in ("power_3d", "p3d", "p3"):
        mapping_dims = {"f": 1, "l": 2, "q": [3], alias: [0.4], "pnd": []}
        assert ConeLayout.from_dims(mapping_dims) == ConeLayout(1, 2, (3,), (0.4,))

    dims.exp = 1
    assert ConeLayout.from_dims(dims) == ConeLayout(1, 2, (3,), (0.25, 0.8), 1)

    for alias in ("exponential", "exp", "ep"):
        mapping_dims = {"f": 1, "l": 2, "q": [3], alias: 2}
        assert ConeLayout.from_dims(mapping_dims) == ConeLayout(1, 2, (3,), (), 2)

    dims.pnd = [[0.25, 0.75]]
    with pytest.raises(ValueError, match="N-dimensional power"):
        ConeLayout.from_dims(dims)

    dims.pnd = []
    dims.psd = [2]
    with pytest.raises(ValueError, match="positive-semidefinite"):
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


def test_power_cone_constraints_are_exact_dcp_and_dnlp_memberships() -> None:
    alpha = 0.25
    layout = ConeLayout(power_3d=(alpha,))
    primal = cp.Variable(3)
    dual = cp.Variable(3)
    primal_constraints = layout.primal_constraints(primal)
    dual_constraints = layout.dual_constraints(dual)
    problem = cp.Problem(cp.Minimize(cp.sum(primal) + cp.sum(dual)), [*primal_constraints, *dual_constraints])

    assert len(primal_constraints) == len(dual_constraints) == 3
    assert problem.is_dcp()
    assert problem.is_dnlp()

    primal.value = np.array([1.0, 1.0, 1.0])
    dual.value = np.array([alpha, 1.0 - alpha, -1.0])
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in primal_constraints)
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in dual_constraints)
    assert float(primal.value @ dual.value) == pytest.approx(0.0, abs=1e-15)

    primal.value = np.array([1.0, 1.0, 1.1])
    dual.value = np.array([alpha, 1.0 - alpha, 1.1])
    assert np.max(primal_constraints[-1].violation()) > 0.0
    assert np.max(dual_constraints[-1].violation()) > 0.0


def test_exponential_cone_constraints_are_exact_dcp_and_dnlp_memberships() -> None:
    layout = ConeLayout(exponential=1)
    primal = cp.Variable(3)
    dual = cp.Variable(3)
    primal_constraints = layout.primal_constraints(primal)
    dual_constraints = layout.dual_constraints(dual)
    problem = cp.Problem(cp.Minimize(cp.sum(primal) + cp.sum(dual)), [*primal_constraints, *dual_constraints])

    assert len(primal_constraints) == len(dual_constraints) == 3
    assert problem.is_dcp()
    assert problem.is_dnlp()
    assert all(constraint.is_dnlp() for constraint in (*primal_constraints, *dual_constraints))

    primal.value = np.array([0.0, 1.0, 1.0])
    dual.value = np.array([-1.0, -1.0, 1.0])
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in primal_constraints)
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in dual_constraints)
    assert float(primal.value @ dual.value) == pytest.approx(0.0, abs=1e-15)

    primal.value = np.array([1.0, 1.0, 1.0])
    dual.value = np.array([-1.0, -2.0, 1.0])
    assert np.max(primal_constraints[-1].violation()) > 0.0
    assert np.max(dual_constraints[-1].violation()) > 0.0


@pytest.mark.parametrize(
    ("primal", "dual"),
    [
        (np.array([-1.0, 0.0, 2.0]), np.array([0.0, 1.0, 2.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 1.0])),
    ],
)
def test_exponential_cone_constraints_include_closure_and_interior(
    primal: np.ndarray,
    dual: np.ndarray,
) -> None:
    layout = ConeLayout(exponential=1)
    primal_variable = cp.Variable(3)
    dual_variable = cp.Variable(3)
    primal_variable.value = primal
    dual_variable.value = dual

    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in layout.primal_constraints(primal_variable))
    assert all(np.max(constraint.violation()) <= 1e-12 for constraint in layout.dual_constraints(dual_variable))


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


def test_power_cone_distance_covers_membership_polar_and_zero_tail_regions() -> None:
    alpha = 0.25
    layout = ConeLayout(power_3d=(alpha,))
    inside = np.array([1.0, 1.0, -1.0])
    dual_inside = np.array([alpha, 1.0 - alpha, 1.0])
    polar = np.array([-alpha, -(1.0 - alpha), 1.0])
    zero_tail = np.array([-2.0, 3.0, 0.0])

    assert layout.primal_distance(inside) == 0.0
    assert layout.dual_distance(dual_inside) == 0.0
    assert layout.primal_distance(polar) == pytest.approx(np.linalg.norm(polar), abs=5e-7)
    assert layout.primal_distance(zero_tail) == pytest.approx(2.0, abs=5e-7)


def test_exponential_cone_distance_covers_projection_regions() -> None:
    layout = ConeLayout(exponential=1)
    inside = np.array([0.0, 1.0, 1.0])
    dual_inside = np.array([-1.0, -1.0, 1.0])
    polar = np.array([1.0, -1.0, -1.0])
    face_region = np.array([-2.0, -3.0, 4.0])

    assert layout.primal_distance(inside) == 0.0
    assert layout.dual_distance(dual_inside) == 0.0
    assert layout.primal_distance(polar) == pytest.approx(np.linalg.norm(polar), abs=5e-7)
    assert layout.primal_distance(face_region) == pytest.approx(3.0, abs=5e-7)


def test_exponential_cone_projection_has_analytic_smooth_boundary_solution() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([math.e + 1.0, 1.0, math.e - 1.0])

    assert layout.primal_distance(point) == pytest.approx(math.sqrt(math.e**2 + 1.0), rel=1e-6)
    assert layout.dual_distance(np.array([1.0, 1.0, -2.0])) == pytest.approx(math.sqrt(5.0), rel=1e-6)


@pytest.mark.parametrize("exponent", [-500, -100, 100, 500])
@pytest.mark.parametrize("kind", ["exponential", "power_3d"])
def test_nonlinear_cone_distances_are_homogeneous_at_binary_scales(kind: str, exponent: int) -> None:
    layout = ConeLayout(exponential=1) if kind == "exponential" else ConeLayout(power_3d=(0.3,))
    point = np.array([0.3, -0.5, 0.7])
    scale = math.ldexp(1.0, exponent)

    assert layout.primal_distance(np.ldexp(point, exponent)) / scale == pytest.approx(
        layout.primal_distance(point), rel=2e-6, abs=1e-9
    )
    assert layout.dual_distance(np.ldexp(point, exponent)) / scale == pytest.approx(
        layout.dual_distance(point), rel=2e-6, abs=1e-9
    )


def test_exponential_cone_ordinary_projection_matches_clarabel_at_binary_scales() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([0.05481268, -1.0, 0.03155354])
    expected_primal = _independent_exponential_cone_distance(point, dual=False)
    expected_dual = _independent_exponential_cone_distance(-point, dual=True)

    for exponent in (-500, 0, 500):
        scale = math.ldexp(1.0, exponent)
        scaled = np.ldexp(point, exponent)
        assert layout.primal_distance(scaled) / scale == pytest.approx(expected_primal, abs=3e-6)
        assert layout.dual_distance(-scaled) / scale == pytest.approx(expected_dual, abs=3e-6)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("exponent", [0, 500])
def test_exponential_face_distance_below_scs_resolution_is_rescaled(exponent: int) -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([-1.0, -4e-7, 1.0])
    scale = math.ldexp(1.0, exponent)

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(np.ldexp(point, exponent))

    assert distance == pytest.approx(4e-7 * scale, rel=2e-3)
    assert statistics.clarabel_retries == 1
    assert statistics.clarabel_accepts == 1
    assert statistics.fallbacks == 0


def test_exact_nonlinear_membership_bypasses_projection_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_solver(*args, **kwargs):
        raise AssertionError("exact members must not invoke a projection solver")

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", unexpected_solver)
    scale = math.ldexp(1.0, 500)
    assert ConeLayout(exponential=1).primal_distance(np.array([0.0, scale, scale])) == 0.0
    assert ConeLayout(exponential=1).dual_distance(np.array([-scale, -scale, scale])) == 0.0
    assert ConeLayout(power_3d=(0.25,)).primal_distance(np.array([scale, scale, scale])) == 0.0
    assert ConeLayout(power_3d=(0.25,)).dual_distance(np.array([0.25, 0.75, 1.0])) == 0.0
    for exponent in (-500, 0, 500):
        assert ConeLayout(power_3d=(0.5,)).primal_distance(np.ldexp(np.array([4.0, 1.0, 2.0]), exponent)) == 0.0
        assert ConeLayout(power_3d=(0.5,)).dual_distance(np.ldexp(np.array([4.0, 1.0, 4.0]), exponent)) == 0.0
        assert ConeLayout(power_3d=(0.25,)).dual_distance(np.ldexp(np.array([0.25, 0.75, 1.0]), exponent)) == 0.0


@pytest.mark.parametrize(
    ("layout", "inside", "outside", "dual"),
    [
        (
            ConeLayout(exponential=1),
            np.array([0.0, math.ldexp(1.0, 500), math.ldexp(1.0, 500)]),
            np.array(
                [
                    0.0,
                    math.ldexp(1.0, 500),
                    math.nextafter(math.ldexp(1.0, 500), 0.0),
                ]
            ),
            False,
        ),
        (
            ConeLayout(exponential=1),
            np.array([-math.ldexp(1.0, 500), -math.ldexp(1.0, 500), math.ldexp(1.0, 500)]),
            np.array(
                [
                    -math.ldexp(1.0, 500),
                    -math.ldexp(1.0, 500),
                    math.nextafter(math.ldexp(1.0, 500), 0.0),
                ]
            ),
            True,
        ),
        (
            ConeLayout(power_3d=(0.5,)),
            np.array([math.ldexp(1.0, 500)] * 3),
            np.array(
                [
                    math.ldexp(1.0, 500),
                    math.ldexp(1.0, 500),
                    math.nextafter(math.ldexp(1.0, 500), math.inf),
                ]
            ),
            False,
        ),
        (
            ConeLayout(power_3d=(0.5,)),
            np.array(
                [
                    math.ldexp(1.0, 500),
                    math.ldexp(1.0, 500),
                    math.ldexp(1.0, 501),
                ]
            ),
            np.array(
                [
                    math.ldexp(1.0, 500),
                    math.ldexp(1.0, 500),
                    math.nextafter(math.ldexp(1.0, 501), math.inf),
                ]
            ),
            True,
        ),
        (
            ConeLayout(power_3d=(float.fromhex("0x1.343db0340757cp-2"),)),
            np.array(
                [
                    float.fromhex("0x1.343db0340757cp-2"),
                    1.0 - float.fromhex("0x1.343db0340757cp-2"),
                    1.0,
                ]
            ),
            np.array(
                [
                    float.fromhex("0x1.2df906382be4fp+580"),
                    float.fromhex("0x1.5e9a146cdbec8p+581"),
                    float.fromhex("0x1.f5969788f1df0p+581"),
                ]
            ),
            True,
        ),
    ],
)
def test_nonlinear_membership_routes_adjacent_outside_points_to_solver(
    monkeypatch: pytest.MonkeyPatch,
    layout: ConeLayout,
    inside: np.ndarray,
    outside: np.ndarray,
    dual: bool,
) -> None:
    calls: list[str] = []

    def failed_solve(requests, *, solver):
        calls.append(solver)
        return [None] * len(requests)

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", failed_solve)
    distance = layout.dual_distance if dual else layout.primal_distance

    assert distance(inside) == 0.0
    assert distance(outside) == math.hypot(*outside)
    assert calls == ["SCS", "CLARABEL"]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "point",
    [
        np.array([1e-300, -1.0, 1.0]),
        np.array([-1.0, 1e-300, -1.0]),
        np.array([0.05, -1.0, 0.9]),
        np.array([1e-310, -1.0, 1e-12]),
    ],
)
def test_exponential_cone_extreme_ratios_return_finite_diagnostics(point: np.ndarray) -> None:
    layout = ConeLayout(exponential=1)

    assert np.isfinite(layout.primal_distance(point))
    assert np.isfinite(layout.dual_distance(point))


@pytest.mark.parametrize("exponent", [-500, 0, 500])
def test_half_power_cone_scaled_axis_projection_has_analytic_distance(exponent: int) -> None:
    layout = ConeLayout(power_3d=(0.5,))
    scale = math.ldexp(1.0, exponent)
    point = np.array([0.0, 0.0, scale])

    assert layout.primal_distance(point) / scale == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-6)
    assert layout.dual_distance(point) / scale == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-6)


def test_lossy_binary_normalization_uses_finite_zero_point_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_solver(*args, **kwargs):
        raise AssertionError("lossy normalization must not reach a projection solver")

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", unexpected_solver)
    large = math.ldexp(1.0, 1000)
    tiny = math.nextafter(0.0, 1.0)
    exponential_point = np.array([-large, 0.0, -tiny])
    power_point = np.array([large, 0.0, tiny])

    assert ConeLayout(exponential=1).primal_distance(exponential_point) == math.hypot(*exponential_point)
    assert ConeLayout(power_3d=(0.5,)).primal_distance(power_point) == math.hypot(*power_point)


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
        ({"exponential": True}, "exponential must be a nonnegative integer"),
        ({"exponential": -1}, "exponential must be a nonnegative integer"),
        ({"exponential": 1.5}, "exponential must be a nonnegative integer"),
        ({"second_order": (1,)}, "dimension at least 2"),
        ({"second_order": 3}, "second_order must be a sequence"),
    ],
)
def test_layout_rejects_invalid_dimensions(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ConeLayout(**kwargs)


@pytest.mark.parametrize(
    "power_3d",
    [
        0.5,
        (False,),
        (0.0,),
        (1.0,),
        (-0.1,),
        (1.1,),
        (np.nan,),
        (np.inf,),
        (1.0 + 0.0j,),
        ("0.5",),
        ([0.5],),
    ],
)
def test_layout_rejects_invalid_power_cone_exponents(power_3d: object) -> None:
    with pytest.raises(ValueError, match="power_3d"):
        ConeLayout(power_3d=power_3d)  # type: ignore[arg-type]


def test_layout_normalizes_real_power_cone_exponents() -> None:
    layout = ConeLayout(power_3d=(Fraction(1, 4), np.float64(0.75)))

    assert layout.power_3d == (0.25, 0.75)
    assert all(type(alpha) is float for alpha in layout.power_3d)


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
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,), power_3d=(0.3,))
    nonnegative_nonfinite = np.array([0.0, nonfinite, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    soc_nonfinite = np.array([0.0, 0.0, 2.0, nonfinite, 0.0, 1.0, 1.0, 0.0])
    power_nonfinite = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 1.0, nonfinite, 0.0])

    assert np.isinf(layout.primal_distance(nonnegative_nonfinite))
    assert np.isinf(layout.dual_distance(nonnegative_nonfinite))
    assert np.isinf(layout.primal_distance(soc_nonfinite))
    assert np.isinf(layout.dual_distance(soc_nonfinite))
    assert np.isinf(layout.primal_distance(power_nonfinite))
    assert np.isinf(layout.dual_distance(power_nonfinite))
    assert np.isinf(soc_distance([2.0, nonfinite]))


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_exponential_blocks_have_infinite_distance(nonfinite: float) -> None:
    layout = ConeLayout(exponential=1)

    for position in range(3):
        point = np.zeros(3)
        point[position] = nonfinite
        assert np.isinf(layout.primal_distance(point))
        assert np.isinf(layout.dual_distance(point))


def test_mixed_nonlinear_blocks_use_one_primary_batch() -> None:
    layout = ConeLayout(exponential=2, power_3d=(0.3, 0.7))
    point = np.array(
        [
            1.0,
            -1.0,
            0.2,
            0.3,
            -0.5,
            0.7,
            -0.2,
            0.5,
            1.0,
            0.4,
            -0.3,
            0.8,
        ]
    )

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(point)

    assert np.isfinite(distance) and distance > 0.0
    assert statistics.scs_batches == 1
    assert statistics.scs_blocks == 4


def test_kkt_rejection_retries_only_the_failed_block(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_solve(requests, *, solver):
        calls.append((solver, len(requests)))
        if solver == "SCS":
            return [np.zeros(3), np.full(3, 0.25)]
        return [np.full(3, 1.0 / 6.0)]

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", fake_solve)
    layout = ConeLayout(exponential=1, power_3d=(0.5,))
    point = np.array([1.0, -1.0, -1.0, 0.0, 0.0, 1.0])

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(point)

    expected = math.hypot(math.sqrt(3.0), math.sqrt(2.0 / 3.0))
    assert distance == pytest.approx(expected)
    assert calls == [("SCS", 2), ("CLARABEL", 1)]
    assert statistics.clarabel_accepts == 1
    assert statistics.fallbacks == 0


def test_projection_failure_uses_saturated_zero_point_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_solve(requests, *, solver):
        return [None] * len(requests)

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", failed_solve)
    layout = ConeLayout(exponential=1)
    ordinary = np.array([1.0, -1.0, -1.0])
    maximum = float(np.finfo(np.float64).max)
    huge = np.array([maximum, -maximum, -maximum])

    with cone_projection._collect_projection_stats() as statistics:
        assert layout.primal_distance(ordinary) == math.hypot(*ordinary)
        assert layout.primal_distance(huge) == maximum

    assert statistics.fallbacks == 2


def test_invalid_projection_candidates_use_zero_point_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid_solve(requests, *, solver):
        return [np.full(3, 10.0)] * len(requests)

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", invalid_solve)
    point = np.array([1.0, -1.0, -1.0])

    assert ConeLayout(exponential=1).primal_distance(point) == math.hypot(*point)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("failure", ["warning", "exception", "unusable_status"])
def test_solver_failures_are_contained_and_use_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    def failed_solve(*args, **kwargs):
        if failure == "warning":
            warnings.warn("synthetic projection warning", UserWarning, stacklevel=2)
        if failure == "exception":
            raise cp.SolverError("synthetic projection failure")
        return 0.0

    monkeypatch.setattr(cp.Problem, "solve", failed_solve)
    point = np.array([1.0, -1.0, -1.0])

    assert ConeLayout(exponential=1).primal_distance(point) == math.hypot(*point)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (math.nextafter(0.0, 1.0), 1.3),
        (1e-17, 1.3),
        (math.nextafter(1.0, 0.0), math.hypot(1.3, 0.45)),
    ],
)
def test_power_endpoint_limits_bypass_solver(alpha: float, expected: float) -> None:
    layout = ConeLayout(power_3d=(alpha,))
    point = np.array([-1.3, 0.9, 0.45])

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(point)

    assert distance == pytest.approx(expected, abs=2e-7)
    assert statistics.endpoint_limits == 1
    assert statistics.scs_batches == 0


@pytest.mark.filterwarnings("error")
def test_zero_power_endpoint_distance_retries_projection_solvers() -> None:
    layout = ConeLayout(power_3d=(1e-9,))

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(np.array([0.0, 1.0, 1.0]))

    assert np.isfinite(distance) and distance > 0.0
    assert statistics.endpoint_limits == 1
    assert statistics.scs_batches == 1
    assert statistics.clarabel_retries == 1
    assert statistics.clarabel_accepts == 1
    assert statistics.fallbacks == 0


def test_zero_power_endpoint_distance_uses_fallback_for_zero_solver_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failed_solve(requests, *, solver):
        calls.append(solver)
        return [request.target.copy() for request in requests]

    monkeypatch.setattr(cone_projection, "_solve_projection_batch", failed_solve)
    point = np.array([0.0, 1.0, 1.0])

    with cone_projection._collect_projection_stats() as statistics:
        distance = ConeLayout(power_3d=(1e-9,)).primal_distance(point)

    assert distance == math.hypot(*point)
    assert calls == ["SCS", "CLARABEL"]
    assert statistics.endpoint_limits == 1
    assert statistics.fallbacks == 1


def test_resolved_power_exponent_uses_projection_solver() -> None:
    layout = ConeLayout(power_3d=(1e-8,))
    point = np.array([-1.3, 0.9, 0.45])

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(point)

    assert np.isfinite(distance) and distance > 0.0
    assert statistics.endpoint_limits == 0
    assert statistics.scs_batches == 1


def _independent_product_cone_distance(point: np.ndarray, *, dual: bool) -> float:
    projected = cp.Variable(12)
    constraints: list[cp.Constraint] = [
        projected[2:5] >= 0,
        cp.SOC(projected[5], projected[6:8]),
        cp.SOC(projected[8], projected[9:12]),
    ]
    if not dual:
        constraints.insert(0, projected[:2] == 0)
    problem = cp.Problem(cp.Minimize(cp.norm(projected - point, 2)), constraints)

    problem.solve(solver=cp.CLARABEL)

    assert problem.status in cp.settings.SOLUTION_PRESENT
    return float(problem.value)


def _independent_power_cone_distance(point: np.ndarray, alpha: float, *, dual: bool) -> float:
    projected = cp.Variable(3)
    if dual:
        constraint = cp.PowCone3D(
            projected[0] / alpha,
            projected[1] / (1.0 - alpha),
            projected[2],
            alpha,
        )
    else:
        constraint = cp.PowCone3D(projected[0], projected[1], projected[2], alpha)
    problem = cp.Problem(cp.Minimize(cp.norm(projected - point, 2)), [constraint])

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )

    assert problem.status in cp.settings.SOLUTION_PRESENT
    return float(problem.value)


def _independent_exponential_cone_distance(point: np.ndarray, *, dual: bool) -> float:
    projected = cp.Variable(3)
    if dual:
        constraint = cp.ExpCone(-projected[1], -projected[0], math.e * projected[2])
    else:
        constraint = cp.ExpCone(projected[0], projected[1], projected[2])
    problem = cp.Problem(cp.Minimize(cp.norm(projected - point, 2)), [constraint])

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )

    assert problem.status in cp.settings.SOLUTION_PRESENT
    return float(problem.value)


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


def test_exponential_cone_distances_match_independent_clarabel_projections() -> None:
    layout = ConeLayout(exponential=1)
    random_points = np.random.default_rng(271828).normal(size=(12, 3))
    extreme_points = np.array(
        [
            [1e-12, -1.0, 0.75],
            [-1.0, 1e-12, -0.5],
            [0.05, -1.0, 0.9],
            [6.41904613e-6, -1.03027122e-7, 1.0],
            [1.0, 0.0, -1.0],
        ]
    )

    for point in np.vstack([random_points, extreme_points]):
        assert layout.primal_distance(point) == pytest.approx(
            _independent_exponential_cone_distance(point, dual=False),
            abs=3e-6,
        )
        assert layout.dual_distance(point) == pytest.approx(
            _independent_exponential_cone_distance(point, dual=True),
            abs=3e-6,
        )


def test_exponential_cone_random_near_faces_match_analytic_and_clarabel_distances() -> None:
    layout = ConeLayout(exponential=1)
    rng = np.random.default_rng(314159)

    for _ in range(6):
        ratio = rng.uniform(-4.0, 4.0)
        y_value = math.exp(rng.uniform(-2.0, 2.0))
        exponential = math.exp(ratio)
        boundary = np.array([ratio * y_value, y_value, y_value * exponential])
        outward = np.array([exponential, exponential * (1.0 - ratio), -1.0])
        outward /= np.linalg.norm(outward)
        expected = 1e-3 * max(1.0, float(np.linalg.norm(boundary)))
        point = boundary + expected * outward

        distance = layout.primal_distance(point)
        assert distance == pytest.approx(expected, rel=2e-6, abs=5e-8)
        assert distance == pytest.approx(
            _independent_exponential_cone_distance(point, dual=False),
            abs=5e-7,
        )

        dual_x = -math.exp(rng.uniform(-2.0, 2.0))
        dual_ratio = rng.uniform(-4.0, 4.0)
        dual_exponential = math.exp(dual_ratio - 1.0)
        dual_boundary = np.array([dual_x, dual_ratio * dual_x, -dual_x * dual_exponential])
        dual_outward = np.array([(dual_ratio - 1.0) * dual_exponential, -dual_exponential, -1.0])
        dual_outward /= np.linalg.norm(dual_outward)
        expected_dual = 1e-3 * max(1.0, float(np.linalg.norm(dual_boundary)))
        dual_point = dual_boundary + expected_dual * dual_outward

        dual_distance = layout.dual_distance(dual_point)
        assert dual_distance == pytest.approx(expected_dual, rel=2e-6, abs=5e-8)
        assert dual_distance == pytest.approx(
            _independent_exponential_cone_distance(dual_point, dual=True),
            abs=5e-7,
        )


@pytest.mark.parametrize("alpha", [0.02, 0.2, 0.5, 0.8, 0.98])
def test_power_cone_distances_match_independent_clarabel_projections(alpha: float) -> None:
    layout = ConeLayout(power_3d=(alpha,))
    points = np.random.default_rng(round(alpha * 10_000)).normal(size=(5, 3))

    for point in points:
        assert layout.primal_distance(point) == pytest.approx(
            _independent_power_cone_distance(point, alpha, dual=False),
            abs=3e-6,
        )
        assert layout.dual_distance(point) == pytest.approx(
            _independent_power_cone_distance(point, alpha, dual=True),
            abs=3e-6,
        )


@pytest.mark.parametrize("alpha", [1e-4, 1.0 - 1e-4])
def test_power_cone_random_near_endpoint_faces_match_analytic_and_clarabel_distances(alpha: float) -> None:
    layout = ConeLayout(power_3d=(alpha,))
    rng = np.random.default_rng(round(alpha * 1_000_000) + 2718)
    complement = 1.0 - alpha
    dual_scale = math.exp(alpha * math.log(alpha) + complement * math.log(complement))

    for _ in range(4):
        x_value, y_value = np.exp(rng.uniform(-2.0, 2.0, size=2))
        product = math.exp(alpha * math.log(x_value) + complement * math.log(y_value))
        boundary = np.array([x_value, y_value, product])
        outward = np.array([-alpha * product / x_value, -complement * product / y_value, 1.0])
        outward /= np.linalg.norm(outward)
        expected = 1e-3 * max(1.0, float(np.linalg.norm(boundary)))
        point = boundary + expected * outward

        distance = layout.primal_distance(point)
        assert distance == pytest.approx(expected, rel=2e-6, abs=5e-8)
        assert distance == pytest.approx(
            _independent_power_cone_distance(point, alpha, dual=False),
            abs=5e-7,
        )

        dual_boundary = np.array([x_value, y_value, product / dual_scale])
        dual_outward = np.array([-alpha * product / x_value, -complement * product / y_value, dual_scale])
        dual_outward /= np.linalg.norm(dual_outward)
        expected_dual = 1e-3 * max(1.0, float(np.linalg.norm(dual_boundary)))
        dual_point = dual_boundary + expected_dual * dual_outward

        dual_distance = layout.dual_distance(dual_point)
        assert dual_distance == pytest.approx(expected_dual, rel=5e-4, abs=5e-6)


def test_power_cone_dual_distance_uses_moreau_decomposition() -> None:
    rng = np.random.default_rng(1138)
    for alpha in (0.001, 0.2, 0.5, 0.95, 0.999):
        layout = ConeLayout(power_3d=(alpha,))
        for point in rng.normal(size=(8, 3)):
            primal_distance = layout.primal_distance(point)
            dual_distance_of_negative = layout.dual_distance(-point)
            assert math.hypot(primal_distance, dual_distance_of_negative) == pytest.approx(
                np.linalg.norm(point),
                rel=2e-6,
                abs=2e-6,
            )


def test_exponential_cone_dual_distance_uses_moreau_decomposition() -> None:
    layout = ConeLayout(exponential=1)
    points = np.vstack(
        [
            np.random.default_rng(161803).normal(size=(20, 3)),
            np.array(
                [
                    [1e-12, -1.0, 0.75],
                    [-1.0, 1e-12, -0.5],
                    [0.05, -1.0, 0.9],
                ]
            ),
        ]
    )

    for point in points:
        primal_distance = layout.primal_distance(point)
        dual_distance_of_negative = layout.dual_distance(-point)
        assert math.hypot(primal_distance, dual_distance_of_negative) == pytest.approx(
            np.linalg.norm(point),
            rel=2e-6,
            abs=2e-6,
        )


@pytest.mark.parametrize("alpha", [1e-8, 1.0 - 1e-8])
def test_power_cone_projection_handles_near_endpoint_exponents(alpha: float) -> None:
    point = np.array([-1.3, 0.9, 0.45])
    reflected = np.array([point[1], point[0], -point[2]])
    layout = ConeLayout(power_3d=(alpha,))
    reflected_layout = ConeLayout(power_3d=(1.0 - alpha,))

    primal_distance = layout.primal_distance(point)
    dual_distance = layout.dual_distance(point)
    assert np.isfinite(primal_distance) and primal_distance > 0.0
    assert np.isfinite(dual_distance) and dual_distance > 0.0
    assert primal_distance == pytest.approx(reflected_layout.primal_distance(reflected), rel=2e-6)
    assert dual_distance == pytest.approx(reflected_layout.dual_distance(reflected), rel=2e-6)
    assert layout.primal_distance(np.array([*point[:2], -point[2]])) == pytest.approx(primal_distance, rel=2e-6)
    assert layout.dual_distance(np.array([*point[:2], -point[2]])) == pytest.approx(dual_distance, rel=2e-6)


@pytest.mark.filterwarnings("error")
def test_power_cone_small_positive_distance_survives_inaccurate_warning() -> None:
    layout = ConeLayout(power_3d=(1e-8,))
    point = np.array([0.0, 1.0, 1.0])

    with cone_projection._collect_projection_stats() as statistics:
        distance = layout.primal_distance(point)
    scaled_distance = layout.primal_distance(0.1 * point)

    assert distance == pytest.approx(1.261e-7, rel=2e-3)
    assert 10.0 * scaled_distance == pytest.approx(distance, rel=2e-3)
    assert statistics.clarabel_retries == 1
    assert statistics.clarabel_accepts == 1
    assert statistics.fallbacks == 0


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("alpha", [1e-17, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)])
def test_power_cone_projection_handles_every_representable_endpoint_exponent(alpha: float) -> None:
    layout = ConeLayout(power_3d=(alpha,))
    point = np.array([-1.3, 0.9, 0.45])
    primal_distance = layout.primal_distance(point)
    dual_distance = layout.dual_distance(point)
    primal_distance_of_negative = layout.primal_distance(-point)
    dual_distance_of_negative = layout.dual_distance(-point)

    distances = (
        primal_distance,
        dual_distance,
        primal_distance_of_negative,
        dual_distance_of_negative,
    )
    assert all(np.isfinite(distance) and distance >= 0.0 for distance in distances)

    if alpha < 0.5:
        expected = 1.3
        expected_of_negative = math.hypot(0.9, 0.45)
    else:
        expected = math.hypot(1.3, 0.45)
        expected_of_negative = 0.9
    assert primal_distance == pytest.approx(expected, abs=2e-7)
    assert dual_distance == pytest.approx(expected, abs=2e-7)
    assert primal_distance_of_negative == pytest.approx(expected_of_negative, abs=2e-7)
    assert dual_distance_of_negative == pytest.approx(expected_of_negative, abs=2e-7)

    assert math.hypot(primal_distance, dual_distance_of_negative) == pytest.approx(
        np.linalg.norm(point),
        rel=2e-6,
    )
    assert math.hypot(dual_distance, primal_distance_of_negative) == pytest.approx(
        np.linalg.norm(point),
        rel=2e-6,
    )

    for exponent in (-500, 500):
        scale = math.ldexp(1.0, exponent)
        scaled = np.ldexp(point, exponent)
        assert layout.primal_distance(scaled) / scale == pytest.approx(primal_distance, rel=2e-6)
        assert layout.dual_distance(scaled) / scale == pytest.approx(dual_distance, rel=2e-6)


def test_mixed_product_distance_aggregates_power_blocks_in_canonical_order() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,), power_3d=(0.5,))
    point = np.array([3.0, -4.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0])

    assert layout.primal_distance(point) == pytest.approx(np.sqrt(28.5), abs=5e-7)
    assert layout.dual_distance(point) == pytest.approx(np.sqrt(19.5), abs=5e-7)


def test_complementarity_uses_unmodified_canonical_order() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,))
    primal = np.array([0.0, 2.0, 3.0, 1.0, 1.0])
    dual = np.array([7.0, 5.0, 4.0, -1.0, 2.0])

    assert layout.complementarity(primal, dual) == pytest.approx(primal @ dual)


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
        _feasibility_tolerance=2e-6,
    )
    source[0] = 99.0
    initial[0] = 99.0

    assert result.epsilon_history == (1e-6,)
    assert result.complementarity == 5e-7
    assert result.succeeded
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
    assert result._feasibility_tolerance == pytest.approx(2e-6)
    with pytest.raises(ValueError):
        result.canonical_primal[0] = 10.0
    with pytest.raises(ValueError):
        selected_run.initial_values["x"][0] = 10.0


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan, True])
def test_result_rejects_invalid_internal_feasibility_tolerance(tolerance: object) -> None:
    with pytest.raises(ValueError, match="feasibility_tolerance"):
        BilevelResult(status="optimal", _feasibility_tolerance=tolerance)  # type: ignore[arg-type]


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
