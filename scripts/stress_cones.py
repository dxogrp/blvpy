"""Run deterministic numerical stress checks for nonlinear cone distances.

This audit is intentionally separate from the default test suite.  The full
preset exercises broad IEEE-754 exponent ranges and native Clarabel
projections; the quick preset is suitable for local iteration.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, localcontext

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from blvpy.cones import ConeLayout

_SEED = 20260924
_SMALLEST_SUBNORMAL = math.nextafter(0.0, 1.0)
_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)

_MOREAU_RTOL = 2e-10
_HOMOGENEITY_RTOL = 5e-12
_ANALYTIC_RTOL = 5e-8
_CLARABEL_ATOL = 5e-6
_CLARABEL_RTOL = 2e-6

_FIXED_POWER_EXPONENTS = (
    math.nextafter(0.0, 1.0),
    1e-300,
    1e-100,
    1e-17,
    1e-8,
    1e-4,
    0.01,
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    0.99,
    1.0 - 1e-4,
    1.0 - 1e-8,
    math.nextafter(1.0, 0.0),
)
_ANALYTIC_POWER_EXPONENTS = (1e-4, 0.02, 0.2, 0.5, 0.8, 0.98, 1.0 - 1e-4)
_CLARABEL_POWER_EXPONENTS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 0.98)


@dataclass(frozen=True, slots=True)
class AuditPreset:
    """Case counts for one stress-audit preset."""

    exponential: int
    exponential_endpoints: int
    power_3d: int
    exponential_clarabel: int
    power_3d_clarabel: int


_PRESETS = {
    "quick": AuditPreset(250, 1_500, 600, 10, 10),
    "full": AuditPreset(5_000, 30_000, 12_000, 100, 100),
}


class AuditFailure(RuntimeError):
    """A deterministic cone stress case violated an audit invariant."""


@dataclass(slots=True)
class ProjectionProblem:
    """Reusable parameterized CVXPY projection problem."""

    point: cp.Parameter
    problem: cp.Problem

    def distance(self, value: NDArray[np.float64]) -> float:
        self.point.value = value
        result = self.problem.solve(
            solver=cp.CLARABEL,
            warm_start=False,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            max_iter=500,
        )
        if (
            self.problem.status not in cp.settings.SOLUTION_PRESENT
            or result is None
            or not math.isfinite(float(result))
        ):
            raise AuditFailure(
                f"Clarabel projection failed with status {self.problem.status!r} and objective {result!r}."
            )
        return math.sqrt(max(float(result), 0.0))


def _rng(seed: int, section: int) -> np.random.Generator:
    """Return an RNG stream isolated from every other audit section."""

    return np.random.default_rng(np.random.SeedSequence([seed, section]))


def _hex_vector(point: NDArray[np.float64]) -> str:
    return "[" + ", ".join(float(entry).hex() for entry in point) + "]"


def _context(
    section: str,
    index: int,
    point: NDArray[np.float64],
    *,
    alpha: float | None = None,
) -> str:
    rendered = f"section={section!r}, case={index}, point={_hex_vector(point)}"
    if alpha is not None:
        rendered += f", alpha={alpha.hex()}"
    return rendered


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _tolerance(expected: float, *, relative: float, ulps: int) -> float:
    if expected == 0.0:
        return 0.0
    return max(relative * abs(expected), ulps * math.ulp(expected))


def _assert_close(
    actual: float,
    expected: float,
    *,
    relative: float,
    ulps: int,
    description: str,
    context: str,
) -> None:
    tolerance = _tolerance(expected, relative=relative, ulps=ulps)
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AuditFailure(
            f"{description}: actual={actual!r}, expected={expected!r}, tolerance={tolerance!r}; {context}"
        )


def _assert_clarabel_close(actual: float, expected: float, *, description: str, context: str) -> None:
    tolerance = _CLARABEL_ATOL + _CLARABEL_RTOL * abs(expected)
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AuditFailure(
            f"{description}: actual={actual!r}, Clarabel={expected!r}, tolerance={tolerance!r}; {context}"
        )


def _random_finite_vector(rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw a finite vector whose Euclidean norm is also representable."""

    point: list[float] = []
    for _ in range(3):
        selector = float(rng.random())
        if selector < 0.08:
            magnitude = 0.0
        elif selector < 0.11:
            magnitude = _SMALLEST_SUBNORMAL
        elif selector < 0.14:
            magnitude = _SMALLEST_NORMAL
        else:
            exponent = int(rng.integers(-1073, 1025))
            # Three entries at this cap still have a finite Euclidean norm.
            upper_mantissa = 0.55 if exponent == 1024 else 1.0
            magnitude = math.ldexp(float(rng.uniform(0.5, upper_mantissa)), exponent)
        point.append(-magnitude if rng.random() < 0.5 else magnitude)
    result = np.asarray(point, dtype=np.float64)
    _require(
        math.isfinite(math.hypot(*result)), f"Stress generator produced a nonrepresentable norm: {_hex_vector(result)}"
    )
    return result


def _random_power_exponent(rng: np.random.Generator, index: int) -> float:
    mode = index % 4
    if mode == 0:
        return _FIXED_POWER_EXPONENTS[(index // 4) % len(_FIXED_POWER_EXPONENTS)]
    if mode == 1:
        exponent = int(rng.integers(-1073, 0))
        return math.ldexp(float(rng.uniform(0.5, 1.0)), exponent)
    if mode == 2:
        exponent = int(rng.integers(-52, 0))
        complement = math.ldexp(float(rng.uniform(0.5, 1.0)), exponent)
        return min(1.0 - complement, math.nextafter(1.0, 0.0))
    return float(rng.uniform(1e-12, 1.0 - 1e-12))


def _scaled_normal_case(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    base = np.asarray(rng.normal(size=3), dtype=np.float64)
    exponent = int(rng.integers(-800, 801))
    point = np.asarray([math.ldexp(float(entry), exponent) for entry in base], dtype=np.float64)
    return point, base, exponent


def _exp_boundary_case(
    rng: np.random.Generator,
    *,
    dual: bool,
    scaled: bool,
) -> tuple[NDArray[np.float64], float]:
    if dual:
        x_value = -math.exp(float(rng.uniform(-2.0, 2.0)))
        ratio = float(rng.uniform(-4.0, 4.0))
        exponential = math.exp(ratio - 1.0)
        boundary = np.array([x_value, ratio * x_value, -x_value * exponential])
        outward = np.array([(ratio - 1.0) * exponential, -exponential, -1.0])
    else:
        ratio = float(rng.uniform(-4.0, 4.0))
        y_value = math.exp(float(rng.uniform(-2.0, 2.0)))
        exponential = math.exp(ratio)
        boundary = np.array([ratio * y_value, y_value, y_value * exponential])
        outward = np.array([exponential, exponential * (1.0 - ratio), -1.0])
    outward /= np.linalg.norm(outward)
    distance = 1e-3 * max(1.0, float(np.linalg.norm(boundary)))
    point = boundary + distance * outward
    if not scaled:
        return point, distance
    exponent = int(rng.integers(-800, 801))
    return (
        np.asarray([math.ldexp(float(entry), exponent) for entry in point]),
        math.ldexp(distance, exponent),
    )


def _power_boundary_case(
    rng: np.random.Generator,
    alpha: float,
    *,
    dual: bool,
    scaled: bool,
) -> tuple[NDArray[np.float64], float]:
    complement = 1.0 - alpha
    x_value, y_value = np.exp(rng.uniform(-2.0, 2.0, size=2))
    product = math.exp(math.fsum((alpha * math.log(x_value), complement * math.log(y_value))))
    if dual:
        dual_scale = math.exp(math.fsum((alpha * math.log(alpha), complement * math.log(complement))))
        boundary = np.array([x_value, y_value, product / dual_scale])
        outward = np.array([-alpha * product / x_value, -complement * product / y_value, dual_scale])
    else:
        boundary = np.array([x_value, y_value, product])
        outward = np.array([-alpha * product / x_value, -complement * product / y_value, 1.0])
    outward /= np.linalg.norm(outward)
    distance = 1e-3 * max(1.0, float(np.linalg.norm(boundary)))
    point = boundary + distance * outward
    if not scaled:
        return point, distance
    exponent = int(rng.integers(-800, 801))
    return (
        np.asarray([math.ldexp(float(entry), exponent) for entry in point]),
        math.ldexp(distance, exponent),
    )


def _distances(layout: ConeLayout, point: NDArray[np.float64]) -> tuple[float, float, float, float]:
    return (
        layout.primal_distance(point),
        layout.dual_distance(-point),
        layout.dual_distance(point),
        layout.primal_distance(-point),
    )


def _check_moreau(
    distances: tuple[float, float, float, float],
    point: NDArray[np.float64],
    *,
    context: str,
) -> None:
    _require(
        all(math.isfinite(distance) and distance >= 0.0 for distance in distances),
        f"A finite point produced an invalid cone distance {distances!r}; {context}",
    )
    norm = math.hypot(*point)
    if norm == 0.0:
        _require(distances == (0.0, 0.0, 0.0, 0.0), f"Zero failed the cone-distance identities; {context}")
        return
    _assert_close(
        math.hypot(distances[0], distances[1]),
        norm,
        relative=_MOREAU_RTOL,
        ulps=8,
        description="Primal/dual Moreau identity failed",
        context=context,
    )
    _assert_close(
        math.hypot(distances[2], distances[3]),
        norm,
        relative=_MOREAU_RTOL,
        ulps=8,
        description="Dual/primal Moreau identity failed",
        context=context,
    )


def _check_homogeneity(
    layout: ConeLayout,
    distances: tuple[float, float, float, float],
    base: NDArray[np.float64],
    exponent: int,
    *,
    context: str,
) -> None:
    for name, actual, reference in (
        ("primal", distances[0], layout.primal_distance(base)),
        ("dual", distances[2], layout.dual_distance(base)),
    ):
        expected = math.ldexp(reference, exponent) if reference else 0.0
        _assert_close(
            actual,
            expected,
            relative=_HOMOGENEITY_RTOL,
            ulps=8,
            description=f"{name.capitalize()} distance homogeneity failed",
            context=context,
        )


def _run_exponential_random(count: int, seed: int) -> None:
    rng = _rng(seed, 1)
    layout = ConeLayout(exponential=1)
    for index in range(count):
        mode = index % 10
        analytic_dual: bool | None = None
        analytic_distance: float | None = None
        base: NDArray[np.float64] | None = None
        exponent: int | None = None
        if mode == 0:
            point, analytic_distance = _exp_boundary_case(rng, dual=False, scaled=True)
            analytic_dual = False
        elif mode == 1:
            point, analytic_distance = _exp_boundary_case(rng, dual=True, scaled=True)
            analytic_dual = True
        elif mode in {2, 3}:
            point, base, exponent = _scaled_normal_case(rng)
        else:
            point = _random_finite_vector(rng)

        context = _context("exponential", index, point)
        distances = _distances(layout, point)
        _check_moreau(distances, point, context=context)
        if analytic_dual is not None and analytic_distance is not None:
            actual = distances[2] if analytic_dual else distances[0]
            _assert_close(
                actual,
                analytic_distance,
                relative=_ANALYTIC_RTOL,
                ulps=16,
                description=f"Analytic EXP {'dual' if analytic_dual else 'primal'} distance failed",
                context=context,
            )
        if base is not None and exponent is not None:
            _check_homogeneity(layout, distances, base, exponent, context=context)


def _first_float_above(value: Decimal) -> float | None:
    candidate = float(value)
    if not math.isfinite(candidate):
        return None
    if Decimal.from_float(candidate) <= value:
        candidate = math.nextafter(candidate, math.inf)
    if not math.isfinite(candidate):
        return None
    while True:
        predecessor = math.nextafter(candidate, -math.inf)
        if Decimal.from_float(predecessor) <= value:
            return candidate
        candidate = predecessor


def _run_exponential_endpoints(count: int, seed: int) -> None:
    layout = ConeLayout(exponential=1)
    magnitude = math.ldexp(1.0, 1000)
    expected = math.ldexp(1.0, -100)
    guards = (
        (False, np.array([-magnitude, 0.0, -expected])),
        (False, np.array([-magnitude, -expected, 0.0])),
        (True, np.array([expected, magnitude, 0.0])),
        (True, np.array([0.0, magnitude, -expected])),
    )
    for index, (dual, point) in enumerate(guards[:count]):
        distance = layout.dual_distance(point) if dual else layout.primal_distance(point)
        _require(
            distance == expected,
            f"Exact EXP face guard returned {distance!r}, expected {expected!r}; "
            f"{_context('exponential-endpoints', index, point)}",
        )
    if count <= len(guards):
        return

    rng = _rng(seed, 2)
    accepted = len(guards)
    attempts = 0
    maximum_attempts = 10 * count
    with localcontext() as context:
        context.prec = 100
        while accepted < count and attempts < maximum_attempts:
            attempts += 1
            y_exponent = int(rng.integers(-900, 901))
            z_exponent = y_exponent + int(rng.integers(-1000, 1001))
            if not -1073 <= z_exponent <= 1023:
                continue
            y_value = math.ldexp(float(rng.uniform(0.5, 1.0)), y_exponent)
            z_value = math.ldexp(float(rng.uniform(0.5, 1.0)), z_exponent)
            decimal_y = Decimal.from_float(y_value)
            decimal_z = Decimal.from_float(z_value)
            dual = (accepted - len(guards)) % 2 == 1
            if dual:
                boundary = decimal_y * (Decimal(1) + (decimal_z / decimal_y).ln())
                candidate = _first_float_above(boundary)
                if candidate is None:
                    continue
                point = np.array([-y_value, -candidate, z_value])
                distance = layout.dual_distance(point)
            else:
                boundary = decimal_y * (decimal_z / decimal_y).ln()
                candidate = _first_float_above(boundary)
                if candidate is None:
                    continue
                point = np.array([candidate, y_value, z_value])
                distance = layout.primal_distance(point)
            case_context = _context("exponential-endpoints", accepted, point)
            _require(
                Decimal.from_float(candidate) > boundary,
                f"Endpoint generator did not produce a strictly outside float; {case_context}",
            )
            _require(
                math.isfinite(distance) and distance > 0.0,
                f"Certified one-ULP EXP point returned invalid distance {distance!r}; {case_context}",
            )
            accepted += 1
    _require(
        accepted == count,
        f"EXP endpoint generator accepted only {accepted:,}/{count:,} cases after {attempts:,} attempts.",
    )


def _run_power_random(count: int, seed: int) -> None:
    rng = _rng(seed, 3)
    for index in range(count):
        mode = index % 12
        analytic_dual: bool | None = None
        analytic_distance: float | None = None
        base: NDArray[np.float64] | None = None
        exponent: int | None = None
        if mode in {0, 1}:
            alpha = _ANALYTIC_POWER_EXPONENTS[(index // 12) % len(_ANALYTIC_POWER_EXPONENTS)]
            analytic_dual = mode == 1
            point, analytic_distance = _power_boundary_case(rng, alpha, dual=analytic_dual, scaled=True)
        else:
            alpha = _random_power_exponent(rng, index)
            if mode in {2, 3}:
                point, base, exponent = _scaled_normal_case(rng)
            else:
                point = _random_finite_vector(rng)

        layout = ConeLayout(power_3d=(alpha,))
        context = _context("power-3d", index, point, alpha=alpha)
        distances = _distances(layout, point)
        _check_moreau(distances, point, context=context)
        if analytic_dual is not None and analytic_distance is not None:
            actual = distances[2] if analytic_dual else distances[0]
            _assert_close(
                actual,
                analytic_distance,
                relative=_ANALYTIC_RTOL,
                ulps=16,
                description=f"Analytic P3D {'dual' if analytic_dual else 'primal'} distance failed",
                context=context,
            )
        if base is not None and exponent is not None:
            _check_homogeneity(layout, distances, base, exponent, context=context)


def _exp_projection_problem(*, dual: bool) -> ProjectionProblem:
    point = cp.Parameter(3)
    projected = cp.Variable(3)
    constraint = (
        cp.ExpCone(-projected[1], -projected[0], math.e * projected[2])
        if dual
        else cp.ExpCone(projected[0], projected[1], projected[2])
    )
    return ProjectionProblem(point, cp.Problem(cp.Minimize(cp.sum_squares(projected - point)), [constraint]))


def _power_projection_problem(alpha: float, *, dual: bool) -> ProjectionProblem:
    point = cp.Parameter(3)
    projected = cp.Variable(3)
    constraint = (
        cp.PowCone3D(projected[0] / alpha, projected[1] / (1.0 - alpha), projected[2], alpha)
        if dual
        else cp.PowCone3D(projected[0], projected[1], projected[2], alpha)
    )
    return ProjectionProblem(point, cp.Problem(cp.Minimize(cp.sum_squares(projected - point)), [constraint]))


def _run_exponential_clarabel(count: int, seed: int) -> int:
    rng = _rng(seed, 4)
    layout = ConeLayout(exponential=1)
    primal_problem = _exp_projection_problem(dual=False)
    dual_problem = _exp_projection_problem(dual=True)
    for index in range(count):
        mode = index % 4
        if mode < 2:
            point = np.asarray(rng.normal(size=3), dtype=np.float64)
        else:
            point, _ = _exp_boundary_case(rng, dual=mode == 3, scaled=False)
        context = _context("exponential-clarabel", index, point)
        _assert_clarabel_close(
            layout.primal_distance(point),
            primal_problem.distance(point),
            description="EXP primal distance disagrees with Clarabel",
            context=context,
        )
        _assert_clarabel_close(
            layout.dual_distance(point),
            dual_problem.distance(point),
            description="EXP dual distance disagrees with Clarabel",
            context=context,
        )
    return 2 * count


def _run_power_clarabel(count: int, seed: int) -> int:
    rng = _rng(seed, 5)
    problems = {
        alpha: (_power_projection_problem(alpha, dual=False), _power_projection_problem(alpha, dual=True))
        for alpha in _CLARABEL_POWER_EXPONENTS
    }
    for index in range(count):
        alpha = _CLARABEL_POWER_EXPONENTS[index % len(_CLARABEL_POWER_EXPONENTS)]
        mode = index % 4
        if mode < 2:
            point = np.asarray(rng.normal(size=3), dtype=np.float64)
        else:
            point, _ = _power_boundary_case(rng, alpha, dual=mode == 3, scaled=False)
        layout = ConeLayout(power_3d=(alpha,))
        primal_problem, dual_problem = problems[alpha]
        context = _context("power-3d-clarabel", index, point, alpha=alpha)
        _assert_clarabel_close(
            layout.primal_distance(point),
            primal_problem.distance(point),
            description="P3D primal distance disagrees with Clarabel",
            context=context,
        )
        _assert_clarabel_close(
            layout.dual_distance(point),
            dual_problem.distance(point),
            description="P3D dual distance disagrees with Clarabel",
            context=context,
        )
    return 2 * count


def _timed_section(label: str, cases: int, action: Callable[[], object]) -> object:
    started = time.perf_counter()
    try:
        result = action()
    except AuditFailure:
        raise
    except Exception as error:
        raise AuditFailure(f"{label} raised {type(error).__name__}: {error}") from error
    elapsed = time.perf_counter() - started
    print(f"{label}: {cases:,} cases passed ({elapsed:.2f}s)", flush=True)
    return result


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(_PRESETS), default="full")
    parser.add_argument("--seed", type=_nonnegative_integer, default=_SEED)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    preset = _PRESETS[options.preset]
    started = time.perf_counter()
    print(f"BLVPY nonlinear-cone stress audit: preset={options.preset}, seed={options.seed}", flush=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="warn"):
                _timed_section(
                    "EXP randomized",
                    preset.exponential,
                    lambda: _run_exponential_random(preset.exponential, options.seed),
                )
                _timed_section(
                    "EXP certified endpoints",
                    preset.exponential_endpoints,
                    lambda: _run_exponential_endpoints(preset.exponential_endpoints, options.seed),
                )
                _timed_section(
                    "P3D randomized",
                    preset.power_3d,
                    lambda: _run_power_random(preset.power_3d, options.seed),
                )
                exp_comparisons = int(
                    _timed_section(
                        "EXP Clarabel",
                        preset.exponential_clarabel,
                        lambda: _run_exponential_clarabel(preset.exponential_clarabel, options.seed),
                    )
                )
                power_comparisons = int(
                    _timed_section(
                        "P3D Clarabel",
                        preset.power_3d_clarabel,
                        lambda: _run_power_clarabel(preset.power_3d_clarabel, options.seed),
                    )
                )
    except (AuditFailure, Warning, FloatingPointError) as error:
        elapsed = time.perf_counter() - started
        print(f"FAILED after {elapsed:.2f}s: {error}", file=sys.stderr, flush=True)
        return 1

    elapsed = time.perf_counter() - started
    random_cases = preset.exponential + preset.exponential_endpoints + preset.power_3d
    clarabel_points = preset.exponential_clarabel + preset.power_3d_clarabel
    print(
        f"PASS: {random_cases:,} randomized/certified cases, {clarabel_points:,} Clarabel points, "
        f"{exp_comparisons + power_comparisons:,} solver comparisons ({elapsed:.2f}s total)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
