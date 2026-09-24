import math
from fractions import Fraction
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

    assert layout.primal_distance(inside) == pytest.approx(0.0, abs=1e-15)
    assert layout.dual_distance(dual_inside) == pytest.approx(0.0, abs=1e-15)
    assert layout.primal_distance(polar) == pytest.approx(np.linalg.norm(polar), rel=1e-14)
    assert layout.primal_distance(zero_tail) == pytest.approx(2.0, rel=1e-14)


def test_exponential_cone_distance_covers_projection_regions() -> None:
    layout = ConeLayout(exponential=1)
    inside = np.array([0.0, 1.0, 1.0])
    dual_inside = np.array([-1.0, -1.0, 1.0])
    polar = np.array([1.0, -1.0, -1.0])
    face_region = np.array([-2.0, -3.0, 4.0])

    assert layout.primal_distance(inside) == pytest.approx(0.0, abs=1e-15)
    assert layout.dual_distance(dual_inside) == pytest.approx(0.0, abs=1e-15)
    assert layout.primal_distance(polar) == pytest.approx(np.linalg.norm(polar), rel=1e-14)
    assert layout.primal_distance(face_region) == pytest.approx(3.0, rel=1e-14)


def test_exponential_cone_projection_has_analytic_smooth_boundary_solution() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([math.e + 1.0, 1.0, math.e - 1.0])

    assert layout.primal_distance(point) == pytest.approx(math.sqrt(math.e**2 + 1.0), rel=2e-14)
    # The dual projection of (1, 1, -2) is (0, 1, 0).
    assert layout.dual_distance(np.array([1.0, 1.0, -2.0])) == pytest.approx(math.sqrt(5.0), rel=2e-14)


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1e100, 1e200])
def test_exponential_cone_distance_is_scale_normalized(scale: float) -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([0.3, -0.5, 0.7])
    primal_reference = layout.primal_distance(point)
    dual_reference = layout.dual_distance(point)

    assert layout.primal_distance(scale * point) / scale == pytest.approx(primal_reference, rel=3e-14)
    assert layout.dual_distance(scale * point) / scale == pytest.approx(dual_reference, rel=3e-14)


def test_exponential_cone_ordinary_projection_matches_clarabel_at_binary_scales() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([0.05481268, -1.0, 0.03155354])
    expected_primal = _independent_exponential_cone_distance(point, dual=False)
    expected_dual = _independent_exponential_cone_distance(-point, dual=True)

    for exponent in (-800, 0, 800):
        scale = math.ldexp(1.0, exponent)
        scaled = np.ldexp(point, exponent)
        assert layout.primal_distance(scaled) / scale == pytest.approx(expected_primal, abs=3e-6)
        assert layout.dual_distance(-scaled) / scale == pytest.approx(expected_dual, abs=3e-6)


def test_exponential_cone_detects_large_exact_boundary_perturbations() -> None:
    layout = ConeLayout(exponential=1)
    scale = math.ldexp(1.0, 40)
    offset = math.ldexp(1.0, -10)
    primal = np.array([offset, scale + offset, scale - offset])
    dual = np.array([-scale, -scale - offset, scale - offset])

    assert layout.primal_distance(primal) == pytest.approx(math.sqrt(3.0) * offset, rel=2e-10)
    assert layout.dual_distance(dual) == pytest.approx(math.sqrt(2.0) * offset, rel=2e-10)


def test_exponential_cone_does_not_round_large_outside_points_into_the_cone() -> None:
    layout = ConeLayout(exponential=1)
    scale = 1e20
    below = np.nextafter(scale, 0.0)

    primal_distance = layout.primal_distance(np.array([0.0, scale, below]))
    dual_distance = layout.dual_distance(np.array([-scale, -scale, below]))

    assert scale - below == 16384.0
    assert np.isfinite(primal_distance) and primal_distance > 0.0
    assert np.isfinite(dual_distance) and dual_distance > 0.0


@pytest.mark.parametrize(
    ("point", "dual", "expected"),
    [
        (
            np.array(
                [
                    float.fromhex("0x1.10ed9a6f354c6p+328"),
                    float.fromhex("0x1.2baca964ee188p+319"),
                    float.fromhex("0x1.f245e54ec9b69p+991"),
                ]
            ),
            False,
            1.725426022327339146e80,
        ),
        (
            np.array(
                [
                    float.fromhex("0x1.fc7e1d08cae00p+310"),
                    float.fromhex("0x1.258b8820e4c98p+305"),
                    float.fromhex("0x1.1fc5e6a2a0856p+385"),
                ]
            ),
            False,
            1.239927639838598852e75,
        ),
        (
            np.array([-20.27415202440371, 188.16580958765275, 168.94569540550697]),
            False,
            2.068487597400123e-15,
        ),
        (
            np.array([-5.888237520874409e-63, 1.8372460860390376e-64, 2.214884604039796e-78]),
            False,
            2.3166589062223175e-93,
        ),
        (
            np.array(
                [
                    -float.fromhex("0x1.294dcbc501016p+2"),
                    float.fromhex("0x1.60429d2f0d4cap+10"),
                    float.fromhex("0x1.2098a38f109a0p-437"),
                ]
            ),
            True,
            9.679274795118902e-146,
        ),
    ],
)
def test_exponential_cone_preserves_one_ulp_smooth_boundary_distances(
    point: np.ndarray,
    dual: bool,
    expected: float,
) -> None:
    layout = ConeLayout(exponential=1)
    distance = layout.dual_distance(point) if dual else layout.primal_distance(point)

    assert distance == pytest.approx(expected, rel=3e-13)


def test_exponential_cone_binary_normalization_preserves_small_face_distances() -> None:
    layout = ConeLayout(exponential=1)
    large = math.ldexp(1.0, 1000)
    expected = math.ldexp(1.0, -100)

    for point in (np.array([-large, 0.0, -expected]), np.array([-large, -expected, 0.0])):
        assert layout.primal_distance(point) == expected
    for point in (np.array([expected, large, 0.0]), np.array([0.0, large, -expected])):
        assert layout.dual_distance(point) == expected


def test_exponential_cone_retries_inexact_subnormal_normalization() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array(
        [
            -float.fromhex("0x1.fbca2e0e69b6cp-77"),
            float.fromhex("0x1.3a5dccc4b2283p+56"),
            -float.fromhex("0x1.af8c8c8ebe758p-998"),
        ]
    )

    assert layout.dual_distance(point) == -point[2]


@pytest.mark.filterwarnings("error")
def test_exponential_cone_endpoint_limits_stay_finite_without_float_warnings() -> None:
    layout = ConeLayout(exponential=1)
    primal_point = np.array([-1.87337558e283, 1.46269072e280, -5.23592150e-105])
    dual_point = np.array([-1.64186439087947524e147, 1.17767475895071370e229, -1.63895105406993598e-121])
    dual_norm_point = np.array([1.03071715e166, -2.88037999e159, 4.71209192e-234])
    dual_positive_endpoint = -np.array([0.05, -1.0, -1e-11])
    dual_zero_tail_endpoint = -np.array([0.038819993662081075, -1.0, -0.0])
    dual_shifted_endpoint = -np.array(
        [
            float.fromhex("0x1.dcef84eaff1d7p-5"),
            -1.0,
            -float.fromhex("0x1.8415cbb6965adp-31"),
        ]
    )
    dual_large_endpoint = -np.array(
        [
            float.fromhex("0x1.449e64107fa9ap+508"),
            -float.fromhex("0x1.5cfbb4074105fp+512"),
            -float.fromhex("0x1.df9000ae7cdd0p+481"),
        ]
    )
    dual_extreme_endpoint = -np.array(
        [
            float.fromhex("0x1.b69b83e9bbd54p+1014"),
            -float.fromhex("0x1.1a2c02ed61a16p+1023"),
            -float.fromhex("0x1.bc1aff575f461p+530"),
        ]
    )
    dual_stagnant_endpoint = -np.array(
        [
            float.fromhex("0x1.83723efd02d42p-10"),
            -1.0,
            -float.fromhex("0x1.58e1d66966279p-1018"),
        ]
    )

    with np.errstate(all="warn"):
        assert layout.primal_distance(primal_point) == 5.2359215e-105
        assert layout.dual_distance(dual_point) == 1.63895105406993598e-121
        assert layout.dual_distance(dual_norm_point) == math.hypot(*dual_norm_point)
        assert layout.dual_distance(dual_positive_endpoint) == pytest.approx(2.791280213955958e-11, rel=3e-14)
        assert layout.dual_distance(dual_zero_tail_endpoint) == pytest.approx(9.276178509796446e-14, rel=3e-14)
        assert layout.dual_distance(dual_shifted_endpoint) == pytest.approx(3.742717013170938e-11, rel=3e-14)
        assert layout.dual_distance(dual_large_endpoint) == pytest.approx(1.542174911535098e144, rel=3e-14)
        assert layout.dual_distance(dual_extreme_endpoint) == pytest.approx(9.769726975428852e161, rel=3e-14)
        assert layout.dual_distance(dual_stagnant_endpoint) == pytest.approx(7.836807490998222e-298, rel=3e-14)


@pytest.mark.filterwarnings("error")
def test_exponential_cone_endpoint_log_corrections_saturate_without_raising() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array(
        [
            -float.fromhex("0x1.93e785b178209p+955"),
            float.fromhex("0x1.475d5716a578cp-68"),
            -float.fromhex("0x1.03b0de439d89bp-719"),
        ]
    )

    assert layout.primal_distance(point) == float.fromhex("0x1.03b0de439d89bp-719")


@pytest.mark.filterwarnings("error")
def test_exponential_cone_positive_endpoint_projection_is_homogeneous() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array(
        [
            float.fromhex("0x1.3934bb2b0d6d1p-207"),
            -float.fromhex("0x1.36384d28cb6b6p-224"),
            float.fromhex("0x1.a43749a1515bep+982"),
        ]
    )
    exponent = -586
    scaled_point = np.array([math.ldexp(float(entry), exponent) for entry in point])

    distance = layout.primal_distance(point)
    scaled_distance = layout.primal_distance(scaled_point)

    assert distance == pytest.approx(7.201296203473927e-66, rel=3e-14)
    assert scaled_distance == math.ldexp(distance, exponent)


@pytest.mark.filterwarnings("error")
def test_exponential_cone_inexact_normalization_retains_valid_normalized_metrics() -> None:
    layout = ConeLayout(exponential=1)
    moreau_point = np.array(
        [
            float.fromhex("0x1.d27c291cf86dcp-996"),
            float.fromhex("0x1.7b2774e8fb780p+400"),
            float.fromhex("0x1.e75bc9140e69ep-207"),
        ]
    )
    moreau_norm = math.hypot(*moreau_point)
    primal_distance = layout.primal_distance(moreau_point)
    opposite_dual_distance = layout.dual_distance(-moreau_point)

    assert math.hypot(primal_distance, opposite_dual_distance) == pytest.approx(moreau_norm, rel=3e-15)

    homogeneous_point = np.array(
        [
            -float.fromhex("0x1.a14fa7342d942p+713"),
            -float.fromhex("0x1.b3b19b17552e0p-551"),
            -float.fromhex("0x1.9263732674008p+188"),
        ]
    )
    exponent = 232
    scaled_point = np.array([math.ldexp(float(entry), exponent) for entry in homogeneous_point])
    distance = layout.dual_distance(homogeneous_point)

    assert layout.dual_distance(scaled_point) == pytest.approx(math.ldexp(distance, exponent), rel=5e-14)


def test_exponential_cone_certified_underflow_remains_strictly_positive() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([-1.0, 0.00134, 0.0])
    distance = layout.primal_distance(point)

    assert distance == math.nextafter(0.0, 1.0)
    assert math.hypot(distance, layout.dual_distance(-point)) == math.hypot(*point)


def test_exponential_cone_logged_displacements_preserve_moreau_norms() -> None:
    layout = ConeLayout(exponential=1)
    point = np.array([-2.59681199641880966e-164, 7.33525122817056057e-40, -9.59394456361389405e251])
    norm = math.hypot(*point)
    primal_distance = layout.primal_distance(point)
    opposite_dual_distance = layout.dual_distance(-point)

    assert primal_distance == norm
    assert math.hypot(primal_distance, opposite_dual_distance) == norm


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "point",
    [
        np.array([1e-300, -1.0, 1.0]),
        np.array([-1.0, 1e-300, -1.0]),
        np.array([1.0, -1e-300, 1.0]),
        np.array([0.05, -1.0, 0.9]),
        np.array([6.41904613e-6, -1.03027122e-7, 1.0]),
        np.array([2.76619574e-140, 1.0, -1.67004426e-121]),
        np.array([1.0, 1.68256427e-185, 8.83517680e-244]),
        np.array([1e-310, -1.0, 1e-12]),
        np.array([-1.0, 1e-310, -1e-3]),
    ],
)
def test_exponential_cone_projection_handles_extreme_ratios(point: np.ndarray) -> None:
    layout = ConeLayout(exponential=1)

    assert np.isfinite(layout.primal_distance(point))
    assert np.isfinite(layout.dual_distance(point))


@pytest.mark.parametrize("scale", [1e-250, 1e-120, 1e120, 1e250])
def test_power_cone_distance_is_scale_normalized(scale: float) -> None:
    layout = ConeLayout(power_3d=(0.3,))
    point = np.array([-0.01, -0.01, 1.0])
    primal_reference = layout.primal_distance(point)
    dual_reference = layout.dual_distance(point)

    assert layout.primal_distance(scale * point) / scale == pytest.approx(primal_reference, rel=2e-13)
    assert layout.dual_distance(scale * point) / scale == pytest.approx(dual_reference, rel=2e-13)


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
    problem = cp.Problem(cp.Minimize(cp.sum_squares(projected - point)), [constraint])

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )

    assert problem.status in cp.settings.SOLUTION_PRESENT
    return float(np.sqrt(max(float(problem.value), 0.0)))


def _independent_exponential_cone_distance(point: np.ndarray, *, dual: bool) -> float:
    projected = cp.Variable(3)
    if dual:
        constraint = cp.ExpCone(-projected[1], -projected[0], math.e * projected[2])
    else:
        constraint = cp.ExpCone(projected[0], projected[1], projected[2])
    problem = cp.Problem(cp.Minimize(cp.sum_squares(projected - point)), [constraint])

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )

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


def test_exponential_cone_distances_match_independent_clarabel_projections() -> None:
    layout = ConeLayout(exponential=1)
    random_points = np.random.default_rng(271828).normal(size=(40, 3))
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
        assert distance == pytest.approx(expected, rel=3e-9, abs=1e-12)
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
        assert dual_distance == pytest.approx(expected_dual, rel=3e-9, abs=1e-12)
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


def test_power_cone_dual_distance_uses_moreau_decomposition() -> None:
    rng = np.random.default_rng(1138)
    for alpha in (0.001, 0.2, 0.5, 0.95, 0.999):
        layout = ConeLayout(power_3d=(alpha,))
        for point in rng.normal(size=(8, 3)):
            primal_distance = layout.primal_distance(point)
            dual_distance_of_negative = layout.dual_distance(-point)
            assert math.hypot(primal_distance, dual_distance_of_negative) == pytest.approx(
                np.linalg.norm(point),
                rel=2e-12,
                abs=2e-12,
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
            rel=2e-10,
            abs=2e-10,
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
    assert primal_distance == pytest.approx(reflected_layout.primal_distance(reflected), rel=2e-9)
    assert dual_distance == pytest.approx(reflected_layout.dual_distance(reflected), rel=2e-9)
    assert layout.primal_distance(np.array([*point[:2], -point[2]])) == pytest.approx(primal_distance, rel=1e-13)
    assert layout.dual_distance(np.array([*point[:2], -point[2]])) == pytest.approx(dual_distance, rel=1e-13)


def test_mixed_product_distance_aggregates_power_blocks_in_canonical_order() -> None:
    layout = ConeLayout(zero=1, nonnegative=1, second_order=(3,), power_3d=(0.5,))
    point = np.array([3.0, -4.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0])

    assert layout.primal_distance(point) == pytest.approx(np.sqrt(28.5), rel=1e-14)
    assert layout.dual_distance(point) == pytest.approx(np.sqrt(19.5), rel=1e-14)


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
