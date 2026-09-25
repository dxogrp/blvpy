"""Shared floating-point primitives for nonlinear cone diagnostics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

_FLOAT_EPSILON = np.finfo(np.float64).eps
_MAX_FLOAT = float(np.finfo(np.float64).max)


def _binary_normalize(vector: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    """Scale a finite vector by an exact common power of two."""

    magnitude = float(np.max(np.abs(vector)))
    if magnitude == 0.0:
        return vector.copy(), 0
    _, exponent = math.frexp(magnitude)
    with np.errstate(under="ignore"):
        return np.ldexp(vector, -exponent), exponent


def _binary_rescale(value: float, exponent: int) -> float:
    """Undo :func:`_binary_normalize`, saturating unrepresentable results."""

    if value == 0.0:
        return 0.0
    try:
        result = math.ldexp(value, exponent)
    except OverflowError:
        return _MAX_FLOAT
    return result if math.isfinite(result) else _MAX_FLOAT


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


def _finite_norm(vector: NDArray[np.float64]) -> float:
    """Return a finite, saturated Euclidean norm for a finite vector."""

    result = math.hypot(*(float(entry) for entry in vector))
    return result if math.isfinite(result) else _MAX_FLOAT


def _saturated_hypot(first: float, second: float) -> float:
    """Combine nonnegative distances without returning overflow infinity."""

    if not math.isfinite(first) or not math.isfinite(second):
        return math.hypot(first, second)
    result = math.hypot(first, second)
    return result if math.isfinite(result) else _MAX_FLOAT


def _log_ratio(numerator: float, denominator: float) -> float:
    """Evaluate ``log(numerator / denominator)`` without ratio overflow."""

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
