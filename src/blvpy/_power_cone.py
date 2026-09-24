"""Numerical distances and projections for three-dimensional power cones."""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal, localcontext

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from ._cone_numeric import (
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


def _power_3d_dual_scale(alpha: float) -> float:
    """Return the tail scale in the dual 3D power-cone inequality."""

    return math.exp(_power_3d_log_dual_scale(alpha))


def _power_3d_log_dual_scale(alpha: float) -> float:
    """Return the log tail scale in the dual power-cone inequality."""

    complement = 1.0 - alpha
    return math.fsum((alpha * math.log(alpha), complement * math.log(complement)))


def _power_3d_distance(
    vector: NDArray[np.float64],
    alpha: float,
    *,
    dual: bool,
) -> float:
    """Return Euclidean distance to one primal or dual power cone."""

    if not np.all(np.isfinite(vector)):
        return float("inf")
    x_head, y_head, z_head = (float(entry) for entry in vector)
    sign_lower_bound = math.hypot(min(x_head, 0.0), min(y_head, 0.0))

    def finish_raw_distance(return_distance: float) -> float:
        if return_distance != 0.0:
            return max(return_distance, sign_lower_bound)
        if dual:
            raw_is_member = _in_power_3d_polar(-x_head, -y_head, abs(z_head), alpha)
        else:
            raw_is_member = _in_power_3d_primal(x_head, y_head, abs(z_head), alpha)
        if raw_is_member:
            return 0.0
        return max(_SMALLEST_SUBNORMAL, sign_lower_bound)

    normalized, exponent = _binary_normalize(vector)
    if not np.any(normalized):
        if dual:
            _, _, return_distance = _power_3d_projection_metrics(-vector, alpha)
        else:
            _, return_distance, _ = _power_3d_projection_metrics(vector, alpha)
        return finish_raw_distance(return_distance)
    normalization_is_exact = _binary_normalization_is_exact(vector, normalized, exponent)
    if dual:
        # Moreau: dist(v, K*) = ||projection_K(-v)||. Scaling the first two
        # coordinates describes K* symbolically but is not an isometry.
        _, _, normalized_distance = _power_3d_projection_metrics(-normalized, alpha)
    else:
        _, normalized_distance, _ = _power_3d_projection_metrics(normalized, alpha)
    distance = _binary_rescale(normalized_distance, exponent)
    if normalization_is_exact and math.isfinite(distance) and distance != 0.0:
        return max(distance, sign_lower_bound)

    # A component spanning more than 1074 binary exponents can underflow or
    # lose low subnormal bits during power-of-two normalization. Re-evaluate
    # at the input scale so a representable correction is not understated.
    if dual:
        _, _, return_distance = _power_3d_projection_metrics(-vector, alpha)
    else:
        _, return_distance, _ = _power_3d_projection_metrics(vector, alpha)
    return finish_raw_distance(return_distance)


def _project_power_3d_unit(
    vector: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Project a finite, normally unit-scaled vector onto a 3D power cone."""

    projected, _, _ = _power_3d_projection_metrics(vector, alpha)
    return projected


def _power_3d_projection_metrics(
    vector: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Project a finite vector onto a 3D power cone.

    The nontrivial boundary case is Hien's scalar equation. It is evaluated
    in logarithmic coordinates and solved in a logit parameter with a
    bracketed Brent iteration. The logit parameter continues to distinguish
    roots whose tail coordinate rounds to either zero or its input value.
    The returned metrics are the distance from the input and the projection
    norm, evaluated without subtracting nearly equal rounded coordinates.
    Inputs are normally power-of-two normalized; raw finite inputs are also
    supported to recover components that disappear at unit scale.
    """

    x_head, y_head, z_head = (float(entry) for entry in vector)
    tail = abs(z_head)
    complement = 1.0 - alpha

    if _in_power_3d_primal(x_head, y_head, tail, alpha):
        projected = vector.copy()
        return projected, 0.0, math.hypot(x_head, y_head, z_head)

    if _in_power_3d_polar(x_head, y_head, tail, alpha):
        projected = np.zeros(3, dtype=np.float64)
        return projected, math.hypot(x_head, y_head, z_head), 0.0

    if tail == 0.0:
        return _power_3d_zero_tail_metrics(vector)

    log_tail = math.log(tail)

    def boundary_residual(logit: float) -> float:
        log_fraction = _log_sigmoid(logit)
        log_complement_fraction = _log_sigmoid(-logit)
        if (
            math.fsum((log_tail, log_fraction)) < _LOG_SMALLEST_SUBNORMAL
            or math.fsum((log_tail, log_complement_fraction)) < _LOG_SMALLEST_SUBNORMAL
        ):
            return _power_3d_boundary_residual_decimal(
                x_head,
                y_head,
                tail,
                alpha,
                logit,
            )
        log_x_ratio = _power_3d_projected_head_log_ratio(
            x_head,
            alpha,
            tail,
            log_fraction,
            log_complement_fraction,
        )
        log_y_ratio = _power_3d_projected_head_log_ratio(
            y_head,
            complement,
            tail,
            log_fraction,
            log_complement_fraction,
        )
        terms = (alpha * log_x_ratio, complement * log_y_ratio)
        residual = math.fsum(terms)
        uncertainty = _power_3d_log_sum_uncertainty(
            (log_x_ratio, log_y_ratio),
            terms,
            residual,
            (alpha, complement),
        )
        if abs(residual) <= uncertainty:
            return _power_3d_boundary_residual_decimal(
                x_head,
                y_head,
                tail,
                alpha,
                logit,
            )
        return residual

    left = -1.0
    left_value = boundary_residual(left)
    for _ in range(1024):
        if math.isfinite(left_value) and left_value >= 0.0:
            break
        left *= 2.0
        if not math.isfinite(left):
            return _power_3d_zero_tail_metrics(vector)
        left_value = boundary_residual(left)
        if not (math.isfinite(left_value) and left_value >= 0.0) and _power_3d_endpoint_is_representable_face(
            vector, alpha, left, zero_tail=True
        ):
            return _power_3d_zero_tail_metrics(vector)
    else:
        return _power_3d_zero_tail_metrics(vector)

    right = 1.0
    right_value = boundary_residual(right)
    for _ in range(1024):
        if math.isfinite(right_value) and right_value <= 0.0:
            break
        right *= 2.0
        if not math.isfinite(right):
            return _power_3d_full_tail_metrics(vector)
        right_value = boundary_residual(right)
        if not (math.isfinite(right_value) and right_value <= 0.0) and _power_3d_endpoint_is_representable_face(
            vector, alpha, right, zero_tail=False
        ):
            return _power_3d_full_tail_metrics(vector)
    else:
        return _power_3d_full_tail_metrics(vector)

    try:
        root = brentq(
            boundary_residual,
            left,
            right,
            xtol=_SMALLEST_SUBNORMAL,
            rtol=4.0 * _FLOAT_EPSILON,
            maxiter=256,
        )
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        if abs(left) < abs(right):
            return _power_3d_zero_tail_metrics(vector)
        return _power_3d_full_tail_metrics(vector)

    log_fraction = _log_sigmoid(root)
    log_complement_fraction = _log_sigmoid(-root)
    log_projected_tail = log_tail + log_fraction
    log_tail_reduction = log_tail + log_complement_fraction
    log_x = _power_3d_projected_head_log(
        x_head,
        alpha,
        log_projected_tail,
        log_tail_reduction,
    )
    log_y = _power_3d_projected_head_log(
        y_head,
        complement,
        log_projected_tail,
        log_tail_reduction,
    )
    projected_tail = tail * math.exp(log_fraction)
    tail_reduction = tail * math.exp(log_complement_fraction)
    projected_x, x_displacement = _power_3d_projected_head(
        x_head,
        alpha,
        log_x,
        log_projected_tail,
        log_tail_reduction,
    )
    projected_y, y_displacement = _power_3d_projected_head(
        y_head,
        complement,
        log_y,
        log_projected_tail,
        log_tail_reduction,
    )
    projected = np.array(
        [projected_x, projected_y, math.copysign(projected_tail, z_head)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(projected)):
        if root >= 0.0:
            return _power_3d_full_tail_metrics(vector)
        return _power_3d_zero_tail_metrics(vector)
    distance = math.hypot(x_displacement, y_displacement, tail_reduction)
    projection_norm = math.hypot(projected_x, projected_y, projected_tail)
    return projected, distance, projection_norm


def _power_3d_zero_tail_projection(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the orthant-face limit of a power-cone projection."""

    projected, _, _ = _power_3d_zero_tail_metrics(vector)
    return projected


def _power_3d_zero_tail_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Return a zero-tail projection and cancellation-free norms."""

    x_head, y_head, z_head = (float(entry) for entry in vector)
    projected_x = max(x_head, 0.0)
    projected_y = max(y_head, 0.0)
    projected = np.array([projected_x, projected_y, 0.0], dtype=np.float64)
    distance = math.hypot(min(x_head, 0.0), min(y_head, 0.0), z_head)
    projection_norm = math.hypot(projected_x, projected_y)
    return projected, distance, projection_norm


def _power_3d_full_tail_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Return the opposite endpoint limit of a power-cone projection."""

    x_head, y_head, z_head = (float(entry) for entry in vector)
    projected_x = max(x_head, 0.0)
    projected_y = max(y_head, 0.0)
    projected = np.array([projected_x, projected_y, z_head], dtype=np.float64)
    distance = math.hypot(min(x_head, 0.0), min(y_head, 0.0))
    projection_norm = math.hypot(projected_x, projected_y, z_head)
    return projected, distance, projection_norm


def _power_3d_endpoint_is_representable_face(
    vector: NDArray[np.float64],
    alpha: float,
    logit: float,
    *,
    zero_tail: bool,
) -> bool:
    """Certify that all corrections beyond an endpoint face round to zero."""

    x_head, y_head, z_head = (float(entry) for entry in vector)
    tail = abs(z_head)
    log_tail = math.log(tail)
    log_fraction = _log_sigmoid(logit)
    log_complement_fraction = _log_sigmoid(-logit)
    log_projected_tail = math.fsum((log_tail, log_fraction))
    log_tail_reduction = math.fsum((log_tail, log_complement_fraction))
    endpoint_log = log_projected_tail if zero_tail else log_tail_reduction
    if endpoint_log >= _LOG_SMALLEST_SUBNORMAL:
        return False

    for head, weight in ((x_head, alpha), (y_head, 1.0 - alpha)):
        if head > 0.0:
            change_log = _power_3d_projected_head_displacement_log(
                head,
                weight,
                log_projected_tail,
                log_tail_reduction,
            )
        else:
            change_log = _power_3d_projected_head_log(
                head,
                weight,
                log_projected_tail,
                log_tail_reduction,
            )
        if change_log >= _LOG_SMALLEST_SUBNORMAL:
            return False
    return True


def _in_power_3d_primal(x: float, y: float, tail: float, alpha: float) -> bool:
    """Test exact floating-point membership in a primal 3D power cone."""

    if x < 0.0 or y < 0.0:
        return False
    if tail == 0.0:
        return True
    if x == 0.0 or y == 0.0:
        return False
    return _power_3d_membership_sign(x, y, tail, alpha, dual=False) >= 0


def _in_power_3d_polar(x: float, y: float, tail: float, alpha: float) -> bool:
    """Test exact floating-point membership in the polar 3D power cone."""

    if x > 0.0 or y > 0.0:
        return False
    if tail == 0.0:
        return True
    if x == 0.0 or y == 0.0:
        return False
    return _power_3d_membership_sign(-x, -y, tail, alpha, dual=True) >= 0


def _power_3d_membership_sign(
    x: float,
    y: float,
    tail: float,
    alpha: float,
    *,
    dual: bool,
) -> int:
    """Return the sign of a power-cone log slack, resolving ambiguity."""

    complement = 1.0 - alpha
    log_x_ratio = _log_ratio(x, tail)
    log_y_ratio = _log_ratio(y, tail)
    logs = [log_x_ratio, log_y_ratio]
    log_errors = [
        _log_ratio_roundoff(x, tail, log_x_ratio),
        _log_ratio_roundoff(y, tail, log_y_ratio),
    ]
    weights = [alpha, complement]
    terms = [alpha * log_x_ratio, complement * log_y_ratio]
    if dual:
        log_alpha = math.log(alpha)
        log_complement = math.log(complement)
        logs.extend((log_alpha, log_complement))
        log_errors.extend((4.0 * math.ulp(log_alpha), 4.0 * math.ulp(log_complement)))
        weights.extend((-alpha, -complement))
        terms.extend((-alpha * log_alpha, -complement * log_complement))
    residual = math.fsum(terms)
    uncertainty = _power_3d_log_sum_uncertainty(
        logs,
        terms,
        residual,
        weights,
        log_errors=log_errors,
    )
    if residual > uncertainty:
        return 1
    if residual < -uncertainty:
        return -1

    precision = _power_3d_decimal_precision(alpha)
    with localcontext() as context:
        context.prec = precision
        decimal_alpha = Decimal.from_float(alpha)
        decimal_complement = Decimal(1) - decimal_alpha
        decimal_x = Decimal.from_float(x)
        decimal_y = Decimal.from_float(y)
        decimal_tail = Decimal.from_float(tail)
        decimal_residual = decimal_alpha * (decimal_x.ln() - decimal_tail.ln())
        decimal_residual += decimal_complement * (decimal_y.ln() - decimal_tail.ln())
        if dual:
            decimal_residual -= decimal_alpha * decimal_alpha.ln()
            decimal_residual -= decimal_complement * decimal_complement.ln()
    return (decimal_residual > 0) - (decimal_residual < 0)


def _power_3d_log_sum_uncertainty(
    logs: Sequence[float],
    terms: Sequence[float],
    residual: float,
    weights: Sequence[float],
    *,
    log_errors: Sequence[float] | None = None,
) -> float:
    """Bound ordinary rounding in a weighted sum of logarithms."""

    contributions = [math.ulp(residual)]
    if log_errors is None:
        log_errors = tuple(4.0 * math.ulp(log_value) for log_value in logs)
    for log_error, term, weight in zip(log_errors, terms, weights, strict=True):
        contributions.append(abs(weight) * log_error)
        contributions.append(2.0 * math.ulp(term))
    return math.fsum(contributions)


def _power_3d_decimal_precision(alpha: float) -> int:
    """Return enough decimal digits to retain the smallest cone weight."""

    smallest_weight = min(alpha, 1.0 - alpha)
    return max(80, math.ceil(-math.log10(smallest_weight)) + 50)


def _power_3d_boundary_residual_decimal(
    x_head: float,
    y_head: float,
    tail: float,
    alpha: float,
    logit: float,
) -> float:
    """Evaluate Hien's weighted log-ratio residual beyond float precision."""

    precision = _power_3d_decimal_precision(alpha)
    with localcontext() as context:
        context.prec = precision
        context.Emin = -999_999_999
        context.Emax = 999_999_999
        one = Decimal(1)
        decimal_alpha = Decimal.from_float(alpha)
        decimal_complement = one - decimal_alpha
        decimal_tail = Decimal.from_float(tail)
        decimal_logit = Decimal.from_float(logit)
        if logit >= 0.0:
            exponential = (-decimal_logit).exp()
            denominator = one + exponential
            fraction = one / denominator
            complement_fraction = exponential / denominator
        else:
            exponential = decimal_logit.exp()
            denominator = one + exponential
            fraction = exponential / denominator
            complement_fraction = one / denominator
        projected_tail = decimal_tail * fraction
        tail_reduction = decimal_tail * complement_fraction

        def projected_head(head: float, weight: Decimal) -> Decimal:
            decimal_head = Decimal.from_float(head)
            discriminant_root = (
                decimal_head * decimal_head + Decimal(4) * weight * projected_tail * tail_reduction
            ).sqrt()
            if head < 0.0:
                return Decimal(2) * weight * projected_tail * tail_reduction / (discriminant_root - decimal_head)
            return (decimal_head + discriminant_root) / Decimal(2)

        projected_x = projected_head(x_head, decimal_alpha)
        projected_y = projected_head(y_head, decimal_complement)
        residual = decimal_alpha * (projected_x.ln() - projected_tail.ln())
        residual += decimal_complement * (projected_y.ln() - projected_tail.ln())
    return float(residual)


def _power_3d_projected_head_log_ratio(
    head: float,
    weight: float,
    tail: float,
    log_fraction: float,
    log_complement_fraction: float,
) -> float:
    """Evaluate the boundary residual's head-to-tail log ratio."""

    log_tail = math.log(tail)
    log_projected_tail = math.fsum((log_tail, log_fraction))
    log_tail_reduction = math.fsum((log_tail, log_complement_fraction))

    # The compensated quadratic below operates in value space. At raw scales
    # where its squares could overflow, use the logarithmic formula; the
    # caller resolves any cancellation in the weighted sum with Decimal.
    square_limit = math.sqrt(np.finfo(np.float64).max) / 4.0
    if max(abs(head), tail) > square_limit:
        log_projected_head = _power_3d_projected_head_log(
            head,
            weight,
            log_projected_tail,
            log_tail_reduction,
        )
        return math.fsum((log_projected_head, -log_projected_tail))

    # Near u = projected_head / projected_tail = 1, solve directly for
    # v = u - 1. The quadratic's residual at u=1 can be assembled from the
    # smaller endpoint fraction without ever subtracting rounded tails.
    if log_fraction <= log_complement_fraction:
        projected_tail = tail * math.exp(log_fraction)
        endpoint_residual = math.fsum(
            (
                head,
                weight * tail,
                -(weight + 1.0) * projected_tail,
            )
        )
    else:
        tail_reduction = tail * math.exp(log_complement_fraction)
        projected_tail = tail - tail_reduction
        endpoint_residual = math.fsum(
            (
                head,
                -tail,
                (weight + 1.0) * tail_reduction,
            )
        )
    linear_coefficient = 2.0 * projected_tail - head
    if linear_coefficient > 0.0 and abs(endpoint_residual) <= 0.25 * linear_coefficient:
        discriminant = math.fsum(
            (
                linear_coefficient * linear_coefficient,
                4.0 * projected_tail * endpoint_residual,
            )
        )
        if discriminant >= 0.0:
            denominator = linear_coefficient + math.sqrt(discriminant)
            if denominator > 0.0:
                relative_difference = 2.0 * endpoint_residual / denominator
                if relative_difference > -1.0:
                    return math.log1p(relative_difference)

    log_projected_head = _power_3d_projected_head_log(
        head,
        weight,
        log_projected_tail,
        log_tail_reduction,
    )
    return math.fsum((log_projected_head, -log_projected_tail))


def _power_3d_projected_head(
    head: float,
    weight: float,
    log_projected_head: float,
    log_projected_tail: float,
    log_tail_reduction: float,
) -> tuple[float, float]:
    """Return one projected head and its stable displacement."""

    if head <= 0.0:
        projected = _exp_or_zero(log_projected_head)
        return projected, projected - head

    log_displacement = _power_3d_projected_head_displacement_log(
        head,
        weight,
        log_projected_tail,
        log_tail_reduction,
    )
    displacement = _exp_or_zero(log_displacement)
    return head + displacement, displacement


def _power_3d_projected_head_displacement_log(
    head: float,
    weight: float,
    log_projected_tail: float,
    log_tail_reduction: float,
) -> float:
    """Return the log increase in one strictly positive projected head."""

    discriminant_root = _power_3d_discriminant_root(
        head,
        weight,
        log_projected_tail,
        log_tail_reduction,
    )
    return math.fsum(
        (
            math.log(2.0 * weight),
            log_projected_tail,
            log_tail_reduction,
            -_log_sum_positive(discriminant_root, head),
        )
    )


def _power_3d_projected_head_log(
    head: float,
    weight: float,
    log_projected_tail: float,
    log_tail_reduction: float,
) -> float:
    """Evaluate log((head + sqrt(head**2 + 4*weight*q))/2)."""

    log_q = log_projected_tail + log_tail_reduction
    discriminant_root = _power_3d_discriminant_root(
        head,
        weight,
        log_projected_tail,
        log_tail_reduction,
    )
    if head > 0.0:
        return _log_sum_positive(head, discriminant_root) - math.log(2.0)
    if head == 0.0:
        return 0.5 * (math.log(weight) + log_q)
    # Rationalize the numerator to avoid cancellation for a negative head.
    return math.log(2.0 * weight) + log_q - math.log(discriminant_root - head)


def _log_sum_positive(first: float, second: float) -> float:
    """Evaluate the log of a sum of positive finite scalars."""

    larger = max(first, second)
    smaller = min(first, second)
    return math.log(larger) + math.log1p(smaller / larger)


def _power_3d_discriminant_root(
    head: float,
    weight: float,
    log_projected_tail: float,
    log_tail_reduction: float,
) -> float:
    """Evaluate the quadratic discriminant root without overflow."""

    log_q = log_projected_tail + log_tail_reduction
    log_sqrt_term = 0.5 * (math.log(4.0 * weight) + log_q)
    sqrt_term = 0.0 if log_sqrt_term < _LOG_SMALLEST_SUBNORMAL else math.exp(log_sqrt_term)
    return math.hypot(head, sqrt_term)


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))
