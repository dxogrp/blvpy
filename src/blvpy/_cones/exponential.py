"""Numerical distances and projections for the exponential cone."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .numeric import (
    _FLOAT_EPSILON,
    _LOG_SMALLEST_SUBNORMAL,
    _SMALLEST_SUBNORMAL,
    _binary_normalization_is_exact,
    _binary_normalize,
    _binary_rescale,
    _exp_or_zero,
    _log_ratio,
    _log_ratio_roundoff,
)


def _decimal_span_precision(*values: float, minimum: int) -> int:
    """Return Decimal precision that covers the binary exponent span."""

    exponents = [math.frexp(abs(value))[1] for value in values if value != 0.0]
    if len(exponents) < 2:
        return minimum
    decimal_span = math.ceil((max(exponents) - min(exponents)) * math.log10(2.0))
    return max(minimum, decimal_span + 80)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide finite scalars, returning a signed infinity on overflow."""

    try:
        return numerator / denominator
    except OverflowError:
        negative = (numerator < 0.0) != (denominator < 0.0)
        return math.copysign(float("inf"), -1.0 if negative else 1.0)


def _exponential_distance(
    vector: NDArray[np.float64],
    *,
    dual: bool,
) -> float:
    """Return Euclidean distance to one primal or dual exponential cone."""

    if not np.all(np.isfinite(vector)):
        return float("inf")
    x_value, y_value, z_value = (float(entry) for entry in vector)
    if not dual and x_value <= 0.0 and y_value <= 0.0:
        return math.hypot(y_value, min(z_value, 0.0))
    if dual and x_value >= 0.0 and y_value >= 0.0:
        return math.hypot(x_value, max(-z_value, 0.0))

    if dual:
        sign_lower_bound = math.hypot(max(x_value, 0.0), min(z_value, 0.0))
    else:
        sign_lower_bound = math.hypot(min(y_value, 0.0), min(z_value, 0.0))

    def raw_distance(normalized_fallback: float | None = None) -> float:
        if dual:
            if _in_exponential_polar(-x_value, -y_value, -z_value):
                return 0.0
            raw_metrics = _exponential_projection_metrics(-vector)
            raw_candidate = None if raw_metrics is None else raw_metrics[2]
        else:
            if _in_exponential_primal(x_value, y_value, z_value):
                return 0.0
            raw_metrics = _exponential_projection_metrics(vector)
            raw_candidate = None if raw_metrics is None else raw_metrics[1]
        if raw_candidate is not None and math.isfinite(raw_candidate) and raw_candidate != 0.0:
            return max(raw_candidate, sign_lower_bound)
        if normalized_fallback is not None and math.isfinite(normalized_fallback) and normalized_fallback != 0.0:
            return max(normalized_fallback, sign_lower_bound)
        if raw_candidate == 0.0:
            # Exact membership was rejected above, so a zero metric here is a
            # certified positive distance below binary64's representable
            # range.  Preserve the nonmembership contract with the least
            # positive result instead of replacing it by a loose face bound.
            return max(_SMALLEST_SUBNORMAL, sign_lower_bound)
        return max(_exponential_feasible_upper_distance(vector, dual=dual), sign_lower_bound)

    normalized, exponent = _binary_normalize(vector)
    if not np.any(normalized):
        return raw_distance()
    normalization_is_exact = _binary_normalization_is_exact(vector, normalized, exponent)
    if dual:
        # Moreau: dist(v, K*) = ||projection_K(-v)||.  The usual coordinate
        # representation K* = {(-y, -x, e*z) in K} is not an isometry.
        metrics = _exponential_projection_metrics(-normalized)
        if metrics is None:
            return raw_distance()
        _, _, normalized_distance = metrics
    else:
        metrics = _exponential_projection_metrics(normalized)
        if metrics is None:
            return raw_distance()
        _, normalized_distance, _ = metrics
    distance = _binary_rescale(normalized_distance, exponent)
    if normalization_is_exact and math.isfinite(distance) and distance != 0.0:
        return max(distance, sign_lower_bound)

    # Scaling a very small component into the subnormal range can retain a
    # nonzero value while still losing low mantissa bits.  Retry at the input
    # scale whenever the transform did not round-trip exactly, as well as for
    # zero or overflowed normalized results.
    return raw_distance(distance)


def _exponential_feasible_upper_distance(
    vector: NDArray[np.float64],
    *,
    dual: bool,
) -> float:
    """Return a feasible-coordinate distance bound after projection failure."""

    x_value, y_value, z_value = (float(entry) for entry in vector)
    if dual:
        candidates = [math.hypot(x_value, min(y_value, 0.0), min(z_value, 0.0))]
        if x_value < 0.0 and z_value > 0.0:
            with localcontext() as context:
                context.prec = 200
                decimal_x = Decimal.from_float(x_value)
                decimal_y = Decimal.from_float(y_value)
                decimal_z = Decimal.from_float(z_value)
                boundary = decimal_x * (Decimal(1) + decimal_z.ln() - (-decimal_x).ln())
                if decimal_y < boundary:
                    candidates.append(float(boundary - decimal_y))
    else:
        candidates = [math.hypot(max(x_value, 0.0), y_value, min(z_value, 0.0))]
        if y_value > 0.0 and z_value > 0.0:
            with localcontext() as context:
                context.prec = 200
                decimal_x = Decimal.from_float(x_value)
                decimal_y = Decimal.from_float(y_value)
                decimal_z = Decimal.from_float(z_value)
                boundary = decimal_y * (decimal_z.ln() - decimal_y.ln())
                if decimal_x > boundary:
                    candidates.append(float(decimal_x - boundary))
    return min(candidates)


def _project_exponential_unit(vector: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Project a finite, normally unit-scaled vector onto the exponential cone."""

    metrics = _exponential_projection_metrics(vector)
    return None if metrics is None else metrics[0]


def _exponential_projection_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Project a finite vector onto the exponential cone.

    The nontrivial smooth-boundary case uses Friberg's monotonically
    increasing scalar equation in ``rho = x / y``.  Its sign is evaluated
    after logarithmic scaling, so neither ``exp(rho)`` nor ``exp(-rho)`` is
    formed during root finding. Inputs are normally power-of-two normalized;
    raw finite inputs are also supported to recover components that disappear
    at unit scale.
    """

    x_value, y_value, z_value = (float(entry) for entry in vector)
    if _in_exponential_primal(x_value, y_value, z_value):
        projected = vector.copy()
        return projected, 0.0, math.hypot(x_value, y_value, z_value)
    if _in_exponential_polar(x_value, y_value, z_value):
        projected = np.zeros(3, dtype=np.float64)
        return projected, math.hypot(x_value, y_value, z_value), 0.0

    # The closest point is on the nonsmooth perspective face throughout this
    # region; this is also the limiting case of the smooth-boundary formula.
    if x_value <= 0.0 and y_value <= 0.0:
        projected_z = max(z_value, 0.0)
        projected = np.array([x_value, 0.0, projected_z], dtype=np.float64)
        distance = math.hypot(y_value, min(z_value, 0.0))
        return projected, distance, math.hypot(x_value, projected_z)

    endpoint_limit = _exponential_endpoint_limit_metrics(vector)
    if endpoint_limit is not None:
        return endpoint_limit

    near_boundary = _exponential_near_boundary_metrics(vector)
    if near_boundary is not None:
        return near_boundary

    lower, upper = _exponential_rho_domain(x_value, y_value)
    if not lower < upper:
        return _exponential_heuristic_projection_metrics(vector)

    def coefficients(rho: float) -> tuple[float, float]:
        rho = float(rho)
        # Factored forms avoid cancellation at finite domain endpoints.
        if x_value == 0.0:
            a_value = y_value
        else:
            a_endpoint = 1.0 - _safe_ratio(y_value, x_value)
            if math.isfinite(a_endpoint):
                if x_value > 0.0:
                    a_value = x_value * (rho - a_endpoint)
                else:
                    a_value = (-x_value) * (a_endpoint - rho)
            else:
                a_value = math.fsum((rho * x_value, y_value, -x_value))

        if y_value == 0.0:
            b_value = x_value
        else:
            b_endpoint = _safe_ratio(x_value, y_value)
            if math.isfinite(b_endpoint):
                if y_value > 0.0:
                    b_value = y_value * (b_endpoint - rho)
                else:
                    b_value = (-y_value) * (rho - b_endpoint)
            else:
                b_value = math.fsum((x_value, -rho * y_value))
        return a_value, b_value

    def boundary_residual(rho: float) -> float:
        rho = float(rho)
        a_value, b_value = coefficients(rho)
        if not a_value > 0.0 or not b_value > 0.0:
            return float("nan")
        log_d = _exponential_log_quadratic(rho)
        signed_logs = [
            (1.0, math.log(a_value) + rho),
            (-1.0, math.log(b_value) - rho),
        ]
        if z_value != 0.0:
            signed_logs.append((-math.copysign(1.0, z_value), log_d + math.log(abs(z_value))))
        largest_log = max(log_value for _, log_value in signed_logs)
        if not math.isfinite(largest_log):
            return float("nan")
        return math.fsum(sign * math.exp(log_value - largest_log) for sign, log_value in signed_logs)

    bracket = _exponential_root_bracket(boundary_residual, lower, upper)
    if bracket is None:
        return _exponential_heuristic_projection_metrics(vector)
    left, right = bracket
    if left == right:
        root = left
    else:
        try:
            root = brentq(
                boundary_residual,
                left,
                right,
                xtol=_SMALLEST_SUBNORMAL,
                rtol=8.0 * _FLOAT_EPSILON,
                maxiter=256,
            )
        except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
            return _exponential_heuristic_projection_metrics(vector)

    def projection_at(rho: float) -> tuple[NDArray[np.float64], float, float] | None:
        a_value, b_value = coefficients(rho)
        if not a_value > 0.0 or not b_value > 0.0:
            return None
        log_d = _exponential_log_quadratic(rho)
        log_y = math.log(a_value) - log_d
        try:
            projected_y = _exp_or_zero(log_y)
            projected_z = _exp_or_zero(math.fsum((rho, log_y)))
            log_b = math.log(b_value)
            x_displacement = _exp_or_zero(log_b - log_d)
            z_displacement = _exp_or_zero(math.fsum((log_b, -rho, -log_d)))
        except OverflowError:
            return None
        projected_x = rho * projected_y
        projected = np.array([projected_x, projected_y, projected_z], dtype=np.float64)
        if not np.all(np.isfinite(projected)):
            return None
        if (x_value <= 0.0 <= projected_x) or (projected_x <= 0.0 <= x_value):
            x_displacement = math.fsum((x_value, -projected_x))
        y_displacement = (1.0 - rho) * x_displacement
        if (y_value <= 0.0 <= projected_y) or (projected_y <= 0.0 <= y_value):
            y_displacement = math.fsum((y_value, -projected_y))
        if (z_value <= 0.0 <= projected_z) or (projected_z <= 0.0 <= z_value):
            z_displacement = math.fsum((projected_z, -z_value))
        distance = math.hypot(x_displacement, y_displacement, z_displacement)
        # The closed-form boundary coordinates independently exponentiate
        # terms that can be enormous and nearly cancel.  Reconstructing them
        # from the KKT displacements retains tiny corrections to a large input
        # and gives a stable norm for the Moreau dual distance.
        stable_projected_x = math.fsum((x_value, -x_displacement))
        stable_projected_y = math.fsum((y_value, -y_displacement))
        stable_projected_z = math.fsum((z_value, z_displacement))
        projection_norm = math.hypot(
            stable_projected_x,
            stable_projected_y,
            stable_projected_z,
        )
        return projected, distance, projection_norm

    # A correctly bracketed root can still land a few ULPs to the wrong side
    # of a near-face KKT condition. Inspect adjacent representable roots and
    # choose the closest candidate whose Moreau decomposition is certified.
    roots = [root]
    lower_neighbor = root
    upper_neighbor = root
    for _ in range(32):
        lower_neighbor = math.nextafter(lower_neighbor, left)
        upper_neighbor = math.nextafter(upper_neighbor, right)
        if lower < lower_neighbor < upper:
            roots.append(lower_neighbor)
        if lower < upper_neighbor < upper:
            roots.append(upper_neighbor)
    candidates = [candidate for rho in roots if (candidate := projection_at(rho)) is not None]
    certified = [candidate for candidate in candidates if _certifies_exponential_projection(vector, candidate[0])]
    if certified:
        return min(certified, key=lambda candidate: candidate[1])
    return _exponential_heuristic_projection_metrics(vector)


def _in_exponential_primal(
    x_value: float,
    y_value: float,
    z_value: float,
    tolerance: float = 0.0,
) -> bool:
    """Test membership in the closed primal EXP cone without exponentiating."""

    x_value, y_value, z_value = float(x_value), float(y_value), float(z_value)
    if y_value < -tolerance or z_value < -tolerance:
        return False
    if y_value <= 0.0:
        return x_value <= tolerance and z_value >= -tolerance
    available_z = z_value + tolerance
    if available_z <= 0.0:
        return False
    if tolerance == 0.0:
        return _exponential_membership_sign(x_value, y_value, z_value, polar=False) >= 0
    boundary_x = y_value * _log_ratio(available_z, y_value)
    return x_value <= boundary_x + tolerance


def _in_exponential_polar(
    x_value: float,
    y_value: float,
    z_value: float,
    tolerance: float = 0.0,
) -> bool:
    """Test membership in the polar of the EXP cone in logarithmic form."""

    x_value, y_value, z_value = float(x_value), float(y_value), float(z_value)
    if x_value < -tolerance or z_value > tolerance:
        return False
    if x_value <= 0.0:
        return y_value <= tolerance and z_value <= tolerance
    available_minus_z = -z_value + tolerance
    if available_minus_z <= 0.0:
        return False
    if tolerance == 0.0:
        return _exponential_membership_sign(x_value, y_value, -z_value, polar=True) >= 0
    boundary_y = x_value * math.fsum((1.0, _log_ratio(available_minus_z, x_value)))
    return y_value <= boundary_y + tolerance


def _exponential_membership_sign(
    first: float,
    second: float,
    positive_third: float,
    *,
    polar: bool,
) -> int:
    """Return an EXP log-slack sign, resolving a rounded boundary."""

    log_ratio = _log_ratio(positive_third, first if polar else second)
    if polar:
        boundary = first * math.fsum((1.0, log_ratio))
        slack = boundary - second
        scale = first
    else:
        boundary = second * log_ratio
        slack = boundary - first
        scale = second
    log_error = _log_ratio_roundoff(
        positive_third,
        first if polar else second,
        log_ratio,
    )
    uncertainty = math.fsum(
        (
            abs(scale) * log_error,
            2.0 * math.ulp(boundary),
            math.ulp(slack),
        )
    )
    if slack > uncertainty:
        return 1
    if slack < -uncertainty:
        return -1

    with localcontext() as context:
        context.prec = 100
        decimal_first = Decimal.from_float(first)
        decimal_second = Decimal.from_float(second)
        decimal_third = Decimal.from_float(positive_third)
        if polar:
            decimal_boundary = decimal_first * (Decimal(1) + decimal_third.ln() - decimal_first.ln())
            decimal_slack = decimal_boundary - decimal_second
        else:
            decimal_boundary = decimal_second * (decimal_third.ln() - decimal_second.ln())
            decimal_slack = decimal_boundary - decimal_first
    return (decimal_slack > 0) - (decimal_slack < 0)


def _exponential_near_boundary_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Return the round-accurate KKT limit next to a smooth EXP face.

    A smooth-boundary displacement can be many orders of magnitude smaller
    than the coordinates themselves.  In that regime no binary64 boundary
    point differs from the input in every required coordinate, so subtracting
    a rounded projection reports a false zero and the scalar root can lose its
    bracket.  Decimal arithmetic retains the exact input floats and evaluates
    the local normal displacement.  The branch is restricted to a tiny
    relative boundary slack, where the omitted curvature term is below the
    precision of the returned binary64 distance.
    """

    x_value, y_value, z_value = (float(entry) for entry in vector)
    relative_limit = 1.0e-12
    with localcontext() as context:
        context.prec = _decimal_span_precision(x_value, y_value, z_value, minimum=200)
        context.Emin = -999_999_999
        context.Emax = 999_999_999
        one = Decimal(1)
        decimal_x = Decimal.from_float(x_value)
        decimal_y = Decimal.from_float(y_value)
        decimal_z = Decimal.from_float(z_value)

        if y_value > 0.0 and z_value > 0.0:
            log_ratio = decimal_z.ln() - decimal_y.ln()
            boundary = decimal_y * log_ratio
            slack = decimal_x - boundary
            scale = max(abs(decimal_x), abs(boundary))
            if slack > 0 and (scale == 0 or slack <= Decimal.from_float(relative_limit) * scale):
                gradient = (one, one - log_ratio, -decimal_y / decimal_z)
                gradient_squared = sum((entry * entry for entry in gradient), Decimal(0))
                multiplier = slack / gradient_squared
                projected_decimal = tuple(
                    entry - multiplier * normal
                    for entry, normal in zip((decimal_x, decimal_y, decimal_z), gradient, strict=True)
                )
                distance = slack / gradient_squared.sqrt()
                projection_norm = sum(
                    (entry * entry for entry in projected_decimal),
                    Decimal(0),
                ).sqrt()
                projected = np.array([float(entry) for entry in projected_decimal], dtype=np.float64)
                if np.all(np.isfinite(projected)):
                    return projected, float(distance), float(projection_norm)

        if x_value > 0.0 and z_value < 0.0:
            positive_third = -decimal_z
            log_ratio = positive_third.ln() - decimal_x.ln()
            boundary = decimal_x * (one + log_ratio)
            slack = decimal_y - boundary
            scale = max(abs(decimal_y), abs(boundary))
            if slack > 0 and (scale == 0 or slack <= Decimal.from_float(relative_limit) * scale):
                gradient = (-log_ratio, one, decimal_x / positive_third)
                gradient_squared = sum((entry * entry for entry in gradient), Decimal(0))
                multiplier = slack / gradient_squared
                projected_decimal = tuple(multiplier * entry for entry in gradient)
                polar_decimal = tuple(
                    entry - projected
                    for entry, projected in zip(
                        (decimal_x, decimal_y, decimal_z),
                        projected_decimal,
                        strict=True,
                    )
                )
                projection_norm = slack / gradient_squared.sqrt()
                distance = sum(
                    (entry * entry for entry in polar_decimal),
                    Decimal(0),
                ).sqrt()
                projected = np.array([float(entry) for entry in projected_decimal], dtype=np.float64)
                if np.all(np.isfinite(projected)):
                    return projected, float(distance), float(projection_norm)
    return None


def _exponential_endpoint_limit_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Certify smooth-face limits beyond binary64's exponential range.

    At either infinite end of the EXP boundary, all non-face corrections can
    round below the smallest subnormal even though the distance itself remains
    representable.  These checks bound the boundary coordinate and each KKT
    correction before returning the corresponding floating-point face limit.
    """

    x_value, y_value, z_value = (float(entry) for entry in vector)
    half_subnormal_log = _LOG_SMALLEST_SUBNORMAL - math.log(2.0)

    if x_value < 0.0 and y_value > 0.0 and z_value <= 0.0:
        rho = _safe_ratio(x_value, y_value)
        if rho == float("-inf"):
            corrections_underflow = True
            log_boundary = float("-inf")
        else:
            log_boundary = math.fsum((math.log(y_value), rho))
            log_tail = float("-inf") if z_value == 0.0 else math.log(-z_value)
            if log_tail == float("-inf"):
                log_distance = log_boundary
            elif log_boundary == float("-inf"):
                log_distance = log_tail
            else:
                larger_log = max(log_tail, log_boundary)
                smaller_log = min(log_tail, log_boundary)
                log_distance = larger_log + math.log1p(math.exp(smaller_log - larger_log))
            # These are logarithms of positive corrections.  At extreme
            # ratios their same-sign sum can lie below binary64's finite
            # range; ordinary addition then saturates to ``-inf``, which is
            # exactly the certified-underflow result needed here.  ``fsum``
            # instead raises on that intermediate overflow.
            log_x_correction = rho + log_distance
            log_y_correction = log_x_correction + math.log1p(-rho)
            corrections_underflow = max(log_x_correction, log_y_correction) < half_subnormal_log
        if corrections_underflow:
            boundary_z = _exp_or_zero(log_boundary)
            projected = np.array([x_value, y_value, boundary_z], dtype=np.float64)
            distance = math.fsum((boundary_z, -z_value))
            return projected, distance, math.hypot(x_value, y_value, boundary_z)

    if x_value > 0.0 and y_value < 0.0:
        with localcontext() as context:
            context.prec = _decimal_span_precision(x_value, y_value, z_value, minimum=200)
            context.Emin = -999_999_999
            context.Emax = 999_999_999
            one = Decimal(1)
            two = Decimal(2)
            decimal_x = Decimal.from_float(x_value)
            decimal_y = Decimal.from_float(y_value)
            decimal_z = Decimal.from_float(z_value)
            endpoint = one - decimal_y / decimal_x

            def residual_at(
                candidate: Decimal,
            ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
                exponential = (-candidate).exp()
                numerator = decimal_z + decimal_x * exponential
                denominator = one + candidate * exponential * exponential
                projected_z = numerator / denominator
                quadratic = candidate * candidate - candidate + one
                residual = decimal_x * (candidate - endpoint) - quadratic * projected_z * exponential
                numerator_derivative = -decimal_x * exponential
                denominator_derivative = exponential * exponential * (one - two * candidate)
                projected_z_derivative = (numerator_derivative * denominator - numerator * denominator_derivative) / (
                    denominator * denominator
                )
                quadratic_derivative = two * candidate - one
                residual_derivative = decimal_x - exponential * (
                    (quadratic_derivative - quadratic) * projected_z + quadratic * projected_z_derivative
                )
                return residual, residual_derivative, exponential, projected_z

            lower = endpoint
            if z_value >= 0.0:
                step = one
                upper = endpoint + step
                for _ in range(16):
                    upper_residual, _, _, _ = residual_at(upper)
                    if upper_residual > 0:
                        break
                    step *= two
                    upper = endpoint + step
                else:
                    upper = endpoint
            else:
                upper = (decimal_x / -decimal_z).ln()

            if z_value > 0.0:
                rho = min(max((decimal_z / decimal_x).ln(), lower), upper)
            else:
                rho = lower
            converged = False
            tolerance = Decimal("1e-80")
            for _ in range(512):
                residual, residual_derivative, exponential, projected_z = residual_at(rho)
                if residual <= 0:
                    lower = rho
                else:
                    upper = rho
                if residual == 0:
                    converged = True
                    break
                if upper - lower <= tolerance * max(one, abs(rho)):
                    rho = (lower + upper) / two
                    converged = True
                    break
                updated = rho - residual / residual_derivative if residual_derivative > 0 else (lower + upper) / two
                if updated == rho:
                    converged = True
                    break
                if not lower < updated < upper:
                    updated = endpoint + (rho * rho - rho + one) * projected_z * exponential / decimal_x
                    if updated == rho:
                        converged = True
                        break
                    if not lower < updated < upper:
                        updated = (lower + upper) / two
                rho = updated

            _, _, exponential, projected_z = residual_at(rho)
            if converged and projected_z > 0:
                projected_y = projected_z * exponential
                projected_x = rho * projected_y
                projected_decimal = (projected_x, projected_y, projected_z)
                displacement_decimal = (
                    decimal_x - projected_x,
                    decimal_y - projected_y,
                    decimal_z - projected_z,
                )
                distance = sum(
                    (entry * entry for entry in displacement_decimal),
                    Decimal(0),
                ).sqrt()
                projection_norm = sum(
                    (entry * entry for entry in projected_decimal),
                    Decimal(0),
                ).sqrt()
                projected = np.array([float(entry) for entry in projected_decimal], dtype=np.float64)
                return projected, float(distance), float(projection_norm)

    if x_value > 0.0 and y_value < 0.0 and z_value >= 0.0:
        ratio = _safe_ratio(-y_value, x_value)
        rho = math.fsum((1.0, ratio)) if math.isfinite(ratio) else float("inf")
        if rho == float("inf"):
            corrections_underflow = True
        else:
            log_rho = math.log(rho)
            log_projected_y = float("-inf") if z_value == 0.0 else math.fsum((math.log(z_value), -rho))
            log_projected_x = math.fsum((log_projected_y, log_rho))
            log_z_correction = math.fsum((math.log(x_value), -rho))
            corrections_underflow = (
                max(
                    log_projected_x,
                    log_projected_y,
                    log_z_correction,
                )
                < half_subnormal_log
            )
        if corrections_underflow:
            projected = np.array([0.0, 0.0, z_value], dtype=np.float64)
            return projected, math.hypot(x_value, y_value), z_value
    return None


def _exponential_heuristic_projection(vector: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Return a certified limiting projection when ``rho`` is unresolvable."""

    metrics = _exponential_heuristic_projection_metrics(vector)
    return None if metrics is None else metrics[0]


def _exponential_heuristic_projection_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Return certified projection metrics when ``rho`` is unresolvable."""

    x_value, y_value, z_value = (float(entry) for entry in vector)
    face = np.array([min(x_value, 0.0), 0.0, max(z_value, 0.0)], dtype=np.float64)
    candidates = [
        (
            face,
            math.hypot(max(x_value, 0.0), y_value, min(z_value, 0.0)),
            math.hypot(float(face[0]), float(face[2])),
        )
    ]
    if y_value > 0.0:
        log_boundary_z = math.fsum((math.log(y_value), _safe_ratio(x_value, y_value)))
        if log_boundary_z <= math.log(np.finfo(np.float64).max):
            boundary_z = _exp_or_zero(log_boundary_z)
            projected_z = max(z_value, boundary_z)
            candidate = np.array([x_value, y_value, projected_z], dtype=np.float64)
            candidates.append(
                (
                    candidate,
                    projected_z - z_value,
                    math.hypot(x_value, y_value, projected_z),
                )
            )

    certified = [
        candidate
        for candidate in candidates
        if candidate[1] > 0.0 and _certifies_exponential_projection(vector, candidate[0])
    ]
    if not certified:
        return None
    return min(certified, key=lambda candidate: candidate[1])


def _certifies_exponential_projection(
    vector: NDArray[np.float64],
    projected: NDArray[np.float64],
) -> bool:
    """Check approximate primal feasibility and Moreau optimality."""

    if not np.any(projected):
        return _in_exponential_polar(*vector)

    certification_tolerance = 2e-10
    polar = vector - projected
    if not _in_exponential_primal(*projected, certification_tolerance):
        return False
    if not _in_exponential_polar(*polar, certification_tolerance):
        return False
    projected_scale = max(abs(float(entry)) for entry in projected)
    polar_scale = max(abs(float(entry)) for entry in polar)
    if projected_scale == 0.0 or polar_scale == 0.0:
        return True

    # Compare the pairing in normalized coordinates.  Computing either the
    # raw dot product or the product of norms can overflow for finite vectors,
    # turning the old ``inf <= inf`` check into a false certificate.
    projected_unit = tuple(float(entry) / projected_scale for entry in projected)
    polar_unit = tuple(float(entry) / polar_scale for entry in polar)
    pairing = math.fsum(
        projected_entry * polar_entry for projected_entry, polar_entry in zip(projected_unit, polar_unit, strict=True)
    )
    if pairing == 0.0:
        return True
    norm_product = math.hypot(*projected_unit) * math.hypot(*polar_unit)
    magnitude_log = math.log(projected_scale) + math.log(polar_scale)
    pairing_log = math.log(abs(pairing)) + magnitude_log
    scale_log = max(0.0, math.log(norm_product) + magnitude_log)
    return pairing_log <= math.log(certification_tolerance) + scale_log


def _exponential_rho_domain(x_value: float, y_value: float) -> tuple[float, float]:
    """Return the open interval where Friberg's two linear factors are positive."""

    lower = float("-inf")
    upper = float("inf")
    if x_value > 0.0:
        lower = max(lower, 1.0 - _safe_ratio(y_value, x_value))
    elif x_value < 0.0:
        upper = min(upper, 1.0 - _safe_ratio(y_value, x_value))
    elif y_value <= 0.0:
        return 0.0, 0.0

    if y_value > 0.0:
        upper = min(upper, _safe_ratio(x_value, y_value))
    elif y_value < 0.0:
        lower = max(lower, _safe_ratio(x_value, y_value))
    elif x_value <= 0.0:
        return 0.0, 0.0
    return lower, upper


def _exponential_root_bracket(
    function: Any,
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    """Bracket the increasing EXP projection equation on an open interval."""

    if lower < 0.0 < upper:
        center = 0.0
    elif math.isfinite(lower) and lower >= 0.0:
        width = upper - lower
        offset = min(1.0, 0.5 * width) if math.isfinite(width) else 1.0
        center = lower + offset
        if center <= lower or center >= upper:
            center = math.nextafter(lower, upper)
    elif math.isfinite(upper) and upper <= 0.0:
        width = upper - lower
        offset = min(1.0, 0.5 * width) if math.isfinite(width) else 1.0
        center = upper - offset
        if center <= lower or center >= upper:
            center = math.nextafter(upper, lower)
    else:
        center = 0.0

    center_value = function(center)
    if not math.isfinite(center_value):
        # Move toward the middle of the open interval until both linear
        # factors are representably positive.
        for _ in range(64):
            if math.isfinite(lower) and math.isfinite(upper):
                center = math.nextafter(center, upper)
            elif math.isfinite(lower):
                center = center + max(1.0, abs(center)) * _FLOAT_EPSILON
            else:
                center = center - max(1.0, abs(center)) * _FLOAT_EPSILON
            center_value = function(center)
            if math.isfinite(center_value):
                break
        else:
            return None
    if center_value == 0.0:
        return center, center

    if center_value > 0.0:
        right = center
        left = center
        step = max(1.0, 0.5 * abs(center))
        for _ in range(1024):
            candidate = left - step
            reached_endpoint = math.isfinite(lower) and candidate <= lower
            if reached_endpoint:
                candidate = left + 0.5 * (lower - left)
                if candidate <= lower or candidate >= left:
                    candidate = math.nextafter(lower, upper)
            if not math.isfinite(candidate):
                return None
            value = function(candidate)
            if math.isfinite(value) and value < 0.0:
                return candidate, right
            if reached_endpoint and candidate == math.nextafter(lower, upper):
                return None
            left = candidate
            step *= 2.0
            if not math.isfinite(step):
                return None
        return None

    left = center
    right = center
    step = max(1.0, 0.5 * abs(center))
    for _ in range(1024):
        candidate = right + step
        reached_endpoint = math.isfinite(upper) and candidate >= upper
        if reached_endpoint:
            candidate = right + 0.5 * (upper - right)
            if candidate <= right or candidate >= upper:
                candidate = math.nextafter(upper, lower)
        if not math.isfinite(candidate):
            return None
        value = function(candidate)
        if math.isfinite(value) and value > 0.0:
            return left, candidate
        if reached_endpoint and candidate == math.nextafter(upper, lower):
            return None
        right = candidate
        step *= 2.0
        if not math.isfinite(step):
            return None
    return None


def _exponential_log_quadratic(rho: float) -> float:
    """Evaluate ``log(rho**2 - rho + 1)`` without overflow."""

    absolute = abs(rho)
    if absolute < 1e150:
        return math.log(rho * (rho - 1.0) + 1.0)
    inverse = 1.0 / rho
    return 2.0 * math.log(absolute) + math.log1p(-inverse + inverse * inverse)
