"""Run deterministic solver-grade stress checks for nonlinear cone distances.

This audit is intentionally separate from the default test suite. It checks
the public product-cone distance API over batched, moderately scaled inputs;
the quick preset is suitable for local iteration.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from blvpy._cones.projection import _collect_projection_stats, _ProjectionStats
from blvpy.cones import ConeLayout

_SEED = 20260924

_EXPONENTIAL_BATCH_SIZE = 250
_POWER_3D_BATCH_SIZE = 500
_HOMOGENEITY_CASES = 32

_MOREAU_TOLERANCE = 5e-6
_HOMOGENEITY_RTOL = 5e-6
_HOMOGENEITY_ATOL = 1e-8
_REFERENCE_ATOL = 5e-6
_REFERENCE_RTOL = 5e-6

_MAIN_POWER_EXPONENTS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 0.98)
_EXTREME_POWER_EXPONENTS = (
    math.nextafter(0.0, 1.0),
    1e-300,
    1e-100,
    1e-17,
    1e-8,
    1.0 - 1e-8,
    math.nextafter(1.0, 0.0),
)


@dataclass(frozen=True, slots=True)
class AuditPreset:
    """Case counts for one stress-audit preset."""

    exponential: int
    power_3d: int
    references: int


_PRESETS = {
    "quick": AuditPreset(250, 600, 10),
    "full": AuditPreset(5_000, 12_000, 100),
}


class AuditFailure(RuntimeError):
    """A deterministic cone stress case violated an audit invariant."""


@dataclass(slots=True)
class ProjectionProblem:
    """Reusable parameterized Clarabel projection problem."""

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
        return float(result)


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


def _assert_close(
    actual: float,
    expected: float,
    *,
    absolute: float,
    relative: float,
    description: str,
    context: str,
) -> None:
    tolerance = absolute + relative * abs(expected)
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AuditFailure(
            f"{description}: actual={actual!r}, expected={expected!r}, tolerance={tolerance!r}; {context}"
        )


def _finite_norm(vector: NDArray[np.float64]) -> float:
    magnitude = float(np.max(np.abs(vector), initial=0.0))
    if magnitude == 0.0:
        return 0.0
    scaled = vector / magnitude
    return magnitude * math.sqrt(float(scaled @ scaled))


def _batches(count: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, count, size):
        yield start, min(size, count - start)


def _exp_boundary_case(
    rng: np.random.Generator,
    *,
    dual: bool,
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
    return np.asarray(boundary + distance * outward, dtype=np.float64), distance


def _exp_primal_member(rng: np.random.Generator) -> NDArray[np.float64]:
    ratio = float(rng.uniform(-3.0, 3.0))
    y_value = math.exp(float(rng.uniform(-1.0, 1.0)))
    return np.array([ratio * y_value, y_value, 1.25 * y_value * math.exp(ratio)])


def _exp_polar_member(rng: np.random.Generator) -> NDArray[np.float64]:
    magnitude = math.exp(float(rng.uniform(-1.0, 1.0)))
    ratio = float(rng.uniform(-3.0, 3.0))
    dual = np.array(
        [
            -magnitude,
            -ratio * magnitude + 0.25 * magnitude,
            magnitude * math.exp(ratio - 1.0),
        ]
    )
    return -dual


def _exponential_points(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    points = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        mode = index % 8
        if mode < 4:
            points[index] = rng.normal(size=3)
        elif mode == 4:
            points[index] = _exp_primal_member(rng)
        elif mode == 5:
            points[index] = _exp_polar_member(rng)
        else:
            points[index], _ = _exp_boundary_case(rng, dual=mode == 7)
    return points


def _power_boundary_case(
    rng: np.random.Generator,
    alpha: float,
    *,
    dual: bool,
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
    return np.asarray(boundary + distance * outward, dtype=np.float64), distance


def _power_primal_member(rng: np.random.Generator, alpha: float) -> NDArray[np.float64]:
    x_value, y_value = np.exp(rng.uniform(-1.0, 1.0, size=2))
    product = math.exp(math.fsum((alpha * math.log(x_value), (1.0 - alpha) * math.log(y_value))))
    return np.array([x_value, y_value, 0.75 * product])


def _power_polar_member(rng: np.random.Generator, alpha: float) -> NDArray[np.float64]:
    x_value, y_value = np.exp(rng.uniform(-1.0, 1.0, size=2))
    complement = 1.0 - alpha
    product = math.exp(math.fsum((alpha * math.log(x_value), complement * math.log(y_value))))
    dual_scale = math.exp(math.fsum((alpha * math.log(alpha), complement * math.log(complement))))
    return -np.array([x_value, y_value, 0.75 * product / dual_scale])


def _power_exponent(rng: np.random.Generator, index: int) -> float:
    if index % 2 == 0:
        return _MAIN_POWER_EXPONENTS[(index // 2) % len(_MAIN_POWER_EXPONENTS)]
    return float(rng.uniform(0.02, 0.98))


def _power_points(
    rng: np.random.Generator,
    start: int,
    count: int,
) -> tuple[NDArray[np.float64], tuple[float, ...]]:
    points = np.empty((count, 3), dtype=np.float64)
    alphas: list[float] = []
    for local_index in range(count):
        index = start + local_index
        alpha = _power_exponent(rng, index)
        alphas.append(alpha)
        mode = index % 8
        if mode < 4:
            points[local_index] = rng.normal(size=3)
        elif mode == 4:
            points[local_index] = _power_primal_member(rng, alpha)
        elif mode == 5:
            points[local_index] = _power_polar_member(rng, alpha)
        else:
            points[local_index], _ = _power_boundary_case(rng, alpha, dual=mode == 7)
    return points, tuple(alphas)


def _distances(layout: ConeLayout, point: NDArray[np.float64]) -> tuple[float, float, float, float]:
    return (
        layout.primal_distance(point),
        layout.dual_distance(-point),
        layout.dual_distance(point),
        layout.primal_distance(-point),
    )


def _check_moreau(
    layout: ConeLayout,
    point: NDArray[np.float64],
    *,
    context: str,
) -> tuple[float, float, float, float]:
    distances = _distances(layout, point)
    _require(
        all(math.isfinite(distance) and distance >= 0.0 for distance in distances),
        f"A finite point produced an invalid cone distance {distances!r}; {context}",
    )
    norm = _finite_norm(point)
    if norm == 0.0:
        _require(distances == (0.0, 0.0, 0.0, 0.0), f"Zero failed cone-distance identities; {context}")
        return distances
    for label, first, second in (
        ("primal/dual", distances[0], distances[1]),
        ("dual/primal", distances[2], distances[3]),
    ):
        ratio = math.hypot(first / norm, second / norm)
        if abs(ratio - 1.0) > _MOREAU_TOLERANCE:
            raise AuditFailure(
                f"{label} Moreau identity failed: normalized norm={ratio!r}, tolerance={_MOREAU_TOLERANCE!r}; {context}"
            )
    return distances


def _run_exponential_random(count: int, seed: int) -> None:
    rng = _rng(seed, 1)
    for batch_index, (start, batch_count) in enumerate(_batches(count, _EXPONENTIAL_BATCH_SIZE)):
        points = _exponential_points(rng, batch_count)
        exponent = int(rng.integers(-40, 41))
        points = np.ldexp(points, exponent)
        layout = ConeLayout(exponential=batch_count)
        flattened = points.reshape(-1)
        _check_moreau(
            layout,
            flattened,
            context=f"section='exponential-random', batch={batch_index}, start={start}, scale=2**{exponent}",
        )


def _run_power_random(count: int, seed: int) -> None:
    rng = _rng(seed, 2)
    for batch_index, (start, batch_count) in enumerate(_batches(count, _POWER_3D_BATCH_SIZE)):
        points, alphas = _power_points(rng, start, batch_count)
        exponent = int(rng.integers(-40, 41))
        points = np.ldexp(points, exponent)
        layout = ConeLayout(power_3d=alphas)
        flattened = points.reshape(-1)
        _check_moreau(
            layout,
            flattened,
            context=f"section='power-random', batch={batch_index}, start={start}, scale=2**{exponent}",
        )


def _check_homogeneity(
    layout: ConeLayout,
    point: NDArray[np.float64],
    *,
    context: str,
) -> None:
    references = (layout.primal_distance(point), layout.dual_distance(point))
    for exponent in (-40, 40):
        scaled = np.ldexp(point, exponent)
        actuals = (layout.primal_distance(scaled), layout.dual_distance(scaled))
        for name, actual, reference in zip(("primal", "dual"), actuals, references, strict=True):
            unscaled = math.ldexp(actual, -exponent) if actual else 0.0
            _assert_close(
                unscaled,
                reference,
                absolute=_HOMOGENEITY_ATOL,
                relative=_HOMOGENEITY_RTOL,
                description=f"{name.capitalize()} distance homogeneity failed",
                context=f"{context}, scale=2**{exponent}",
            )


def _run_homogeneity(seed: int) -> None:
    exp_rng = _rng(seed, 3)
    exp_points = _exponential_points(exp_rng, _HOMOGENEITY_CASES).reshape(-1)
    exp_layout = ConeLayout(exponential=_HOMOGENEITY_CASES)
    _check_moreau(exp_layout, exp_points, context="section='exponential-homogeneity', base")
    _check_homogeneity(exp_layout, exp_points, context="section='exponential-homogeneity'")

    power_rng = _rng(seed, 4)
    power_points, alphas = _power_points(power_rng, 0, _HOMOGENEITY_CASES)
    power_points = power_points.reshape(-1)
    power_layout = ConeLayout(power_3d=alphas)
    _check_moreau(power_layout, power_points, context="section='power-homogeneity', base")
    _check_homogeneity(power_layout, power_points, context="section='power-homogeneity'")


def _exp_projection_problem(*, dual: bool) -> ProjectionProblem:
    point = cp.Parameter(3)
    projected = cp.Variable(3)
    constraint = (
        cp.ExpCone(-projected[1], -projected[0], math.e * projected[2])
        if dual
        else cp.ExpCone(projected[0], projected[1], projected[2])
    )
    return ProjectionProblem(point, cp.Problem(cp.Minimize(cp.norm(projected - point, 2)), [constraint]))


def _power_projection_problem(alpha: float, *, dual: bool) -> ProjectionProblem:
    point = cp.Parameter(3)
    projected = cp.Variable(3)
    constraint = (
        cp.PowCone3D(projected[0] / alpha, projected[1] / (1.0 - alpha), projected[2], alpha)
        if dual
        else cp.PowCone3D(projected[0], projected[1], projected[2], alpha)
    )
    return ProjectionProblem(point, cp.Problem(cp.Minimize(cp.norm(projected - point, 2)), [constraint]))


def _run_exponential_references(count: int, seed: int) -> int:
    rng = _rng(seed, 5)
    layout = ConeLayout(exponential=1)
    primal_problem = _exp_projection_problem(dual=False)
    dual_problem = _exp_projection_problem(dual=True)
    for index in range(count):
        mode = index % 4
        analytic_dual: bool | None = None
        analytic_distance: float | None = None
        if mode < 2:
            point = np.asarray(rng.normal(size=3), dtype=np.float64)
        else:
            analytic_dual = mode == 3
            point, analytic_distance = _exp_boundary_case(rng, dual=analytic_dual)
        context = _context("exponential-reference", index, point)
        distances = _check_moreau(layout, point, context=context)
        for name, actual, expected in (
            ("primal", distances[0], primal_problem.distance(point)),
            ("dual", distances[2], dual_problem.distance(point)),
        ):
            _assert_close(
                actual,
                expected,
                absolute=_REFERENCE_ATOL,
                relative=_REFERENCE_RTOL,
                description=f"EXP {name} distance disagrees with Clarabel",
                context=context,
            )
        if analytic_dual is not None and analytic_distance is not None:
            actual = distances[2] if analytic_dual else distances[0]
            _assert_close(
                actual,
                analytic_distance,
                absolute=_REFERENCE_ATOL,
                relative=_REFERENCE_RTOL,
                description=f"Analytic EXP {'dual' if analytic_dual else 'primal'} distance failed",
                context=context,
            )
    return 2 * count


def _run_power_references(count: int, seed: int) -> int:
    rng = _rng(seed, 6)
    problems = {
        alpha: (_power_projection_problem(alpha, dual=False), _power_projection_problem(alpha, dual=True))
        for alpha in _MAIN_POWER_EXPONENTS
    }
    for index in range(count):
        alpha = _MAIN_POWER_EXPONENTS[index % len(_MAIN_POWER_EXPONENTS)]
        mode = index % 4
        analytic_dual: bool | None = None
        analytic_distance: float | None = None
        if mode < 2:
            point = np.asarray(rng.normal(size=3), dtype=np.float64)
        else:
            analytic_dual = mode == 3
            point, analytic_distance = _power_boundary_case(rng, alpha, dual=analytic_dual)
        layout = ConeLayout(power_3d=(alpha,))
        context = _context("power-reference", index, point, alpha=alpha)
        distances = _check_moreau(layout, point, context=context)
        primal_problem, dual_problem = problems[alpha]
        for name, actual, expected in (
            ("primal", distances[0], primal_problem.distance(point)),
            ("dual", distances[2], dual_problem.distance(point)),
        ):
            _assert_close(
                actual,
                expected,
                absolute=_REFERENCE_ATOL,
                relative=_REFERENCE_RTOL,
                description=f"P3D {name} distance disagrees with Clarabel",
                context=context,
            )
        if analytic_dual is not None and analytic_distance is not None:
            actual = distances[2] if analytic_dual else distances[0]
            _assert_close(
                actual,
                analytic_distance,
                absolute=_REFERENCE_ATOL,
                relative=_REFERENCE_RTOL,
                description=f"Analytic P3D {'dual' if analytic_dual else 'primal'} distance failed",
                context=context,
            )
    return 2 * count


def _run_extreme_power_exponents() -> int:
    points = (
        np.array([1.0, 1.0, 2.0]),
        np.array([-1.0, 2.0, 0.5]),
        np.array([2.0, -1.0, -0.5]),
    )
    count = 0
    for alpha in _EXTREME_POWER_EXPONENTS:
        layout = ConeLayout(power_3d=(alpha,))
        for point in points:
            context = _context("power-extreme-alpha", count, point, alpha=alpha)
            _check_moreau(layout, point, context=context)
            count += 1
    return count


def _stats_values(stats: _ProjectionStats) -> tuple[int, int, int, int, int, int]:
    return (
        stats.scs_batches,
        stats.scs_blocks,
        stats.clarabel_retries,
        stats.clarabel_accepts,
        stats.endpoint_limits,
        stats.fallbacks,
    )


def _stats_delta(before: tuple[int, ...], stats: _ProjectionStats) -> tuple[int, ...]:
    return tuple(after - earlier for earlier, after in zip(before, _stats_values(stats), strict=True))


def _timed_section(
    label: str,
    cases: int,
    action: Callable[[], object],
    stats: _ProjectionStats,
) -> object:
    started = time.perf_counter()
    before = _stats_values(stats)
    try:
        result = action()
    except AuditFailure:
        raise
    except Exception as error:
        raise AuditFailure(f"{label} raised {type(error).__name__}: {error}") from error
    elapsed = time.perf_counter() - started
    scs_batches, scs_blocks, retries, accepts, endpoints, fallbacks = _stats_delta(before, stats)
    print(
        f"{label}: {cases:,} cases passed ({elapsed:.2f}s; SCS {scs_batches:,} batches/{scs_blocks:,} blocks, "
        f"Clarabel {accepts:,}/{retries:,} retries accepted, endpoint limits {endpoints:,}, "
        f"fallbacks {fallbacks:,})",
        flush=True,
    )
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
        with _collect_projection_stats() as stats, warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="warn"):
                _timed_section(
                    "EXP randomized",
                    preset.exponential,
                    lambda: _run_exponential_random(preset.exponential, options.seed),
                    stats,
                )
                _timed_section(
                    "P3D randomized",
                    preset.power_3d,
                    lambda: _run_power_random(preset.power_3d, options.seed),
                    stats,
                )
                _timed_section(
                    "Moderate homogeneity",
                    2 * _HOMOGENEITY_CASES,
                    lambda: _run_homogeneity(options.seed),
                    stats,
                )
                exp_comparisons = int(
                    _timed_section(
                        "EXP Clarabel/analytic references",
                        preset.references,
                        lambda: _run_exponential_references(preset.references, options.seed),
                        stats,
                    )
                )
                power_comparisons = int(
                    _timed_section(
                        "P3D Clarabel/analytic references",
                        preset.references,
                        lambda: _run_power_references(preset.references, options.seed),
                        stats,
                    )
                )
                extreme_cases = int(
                    _timed_section(
                        "P3D extreme-alpha robustness",
                        len(_EXTREME_POWER_EXPONENTS) * 3,
                        _run_extreme_power_exponents,
                        stats,
                    )
                )
    except (AuditFailure, Warning, FloatingPointError) as error:
        elapsed = time.perf_counter() - started
        print(f"FAILED after {elapsed:.2f}s: {error}", file=sys.stderr, flush=True)
        return 1

    elapsed = time.perf_counter() - started
    print(
        f"PASS: {preset.exponential + preset.power_3d:,} randomized blocks, "
        f"{2 * preset.references:,} reference points, {exp_comparisons + power_comparisons:,} comparisons, "
        f"{extreme_cases:,} extreme-alpha cases; seed={options.seed}; "
        f"SCS={stats.scs_batches:,} batches/{stats.scs_blocks:,} blocks, "
        f"Clarabel={stats.clarabel_accepts:,}/{stats.clarabel_retries:,} retries accepted, "
        f"endpoint limits={stats.endpoint_limits:,}, fallbacks={stats.fallbacks:,}; "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
