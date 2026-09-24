"""Shared floating-point primitives for nonlinear cone projections."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

_FLOAT_EPSILON = np.finfo(np.float64).eps
_SMALLEST_SUBNORMAL = math.nextafter(0.0, 1.0)
_LOG_SMALLEST_SUBNORMAL = math.log(_SMALLEST_SUBNORMAL)


def _binary_normalize(vector: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    """Scale a finite vector by an exact power of two.

    Dividing by an arbitrary largest entry can round away a small, but
    representable, distance from a cone boundary. ``frexp``/``ldexp`` avoid
    that extra rounding for components that remain representable while
    keeping the projection problem at unit scale. Callers retry the raw input
    if the exponent span makes a decisive component underflow.
    """

    magnitude = float(np.max(np.abs(vector)))
    if magnitude == 0.0:
        return vector.copy(), 0
    _, exponent = math.frexp(magnitude)
    # A component can legitimately underflow while the largest component is
    # brought to unit scale.  The distance routines retry the unscaled input
    # whenever that loss could hide a representable residual.
    with np.errstate(under="ignore"):
        return np.ldexp(vector, -exponent), exponent


def _binary_rescale(value: float, exponent: int) -> float:
    """Undo :func:`_binary_normalize` without intermediate overflow."""

    if value == 0.0:
        return 0.0
    try:
        result = math.ldexp(value, exponent)
    except OverflowError:
        return float("inf")
    return result if math.isfinite(result) else float("inf")


def _binary_normalization_is_exact(
    vector: NDArray[np.float64],
    normalized: NDArray[np.float64],
    exponent: int,
) -> bool:
    """Return whether every normalized component round-trips exactly."""

    for original, scaled in zip(vector, normalized, strict=True):
        try:
            restored = math.ldexp(float(scaled), exponent)
        except OverflowError:
            return False
        if restored != float(original):
            return False
    return True


def _log_ratio(numerator: float, denominator: float) -> float:
    """Evaluate ``log(numerator / denominator)`` without ratio overflow.

    The ``log1p`` branch also retains a one-ULP separation when the operands
    are close.  Sterbenz's lemma makes the subtraction exact in that branch.
    """

    difference = numerator - denominator
    if abs(difference) <= 0.5 * denominator:
        return math.log1p(difference / denominator)
    return math.fsum((math.log(numerator), -math.log(denominator)))


def _log_ratio_roundoff(numerator: float, denominator: float, result: float) -> float:
    """Estimate outward roundoff for :func:`_log_ratio`."""

    difference = numerator - denominator
    if abs(difference) <= 0.5 * denominator:
        return math.fsum((8.0 * math.ulp(result), 4.0 * _FLOAT_EPSILON * abs(result)))
    log_numerator = math.log(numerator)
    log_denominator = math.log(denominator)
    return math.fsum(
        (
            4.0 * math.ulp(log_numerator),
            4.0 * math.ulp(log_denominator),
            2.0 * math.ulp(result),
        )
    )


def _exp_or_zero(value: float) -> float:
    return 0.0 if value < _LOG_SMALLEST_SUBNORMAL else math.exp(value)
