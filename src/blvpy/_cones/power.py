"""Membership and endpoint helpers for three-dimensional power cones."""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext

import numpy as np
from numpy.typing import NDArray

from .numeric import _log_ratio, _log_ratio_roundoff

_ENDPOINT_RESOLUTION = 1e-7


def _power_3d_dual_scale(alpha: float) -> float:
    """Return the tail scale in the dual 3D power-cone inequality."""

    return math.exp(_power_3d_log_dual_scale(alpha))


def _power_3d_log_dual_scale(alpha: float) -> float:
    """Return the log tail scale in the dual power-cone inequality."""

    complement = 1.0 - alpha
    return math.fsum((alpha * math.log(alpha), complement * math.log(complement)))


def _in_power_3d_primal(
    x: float,
    y: float,
    tail: float,
    alpha: float,
    tolerance: float = 0.0,
) -> bool:
    """Test membership in a primal 3D power cone."""

    if tolerance != 0.0:
        if x < -tolerance or y < -tolerance:
            return False
        available_tail = max(abs(tail) - tolerance, 0.0)
        if available_tail == 0.0:
            return True
        available_x = x + tolerance
        available_y = y + tolerance
        if available_x <= 0.0 or available_y <= 0.0:
            return False
        complement = 1.0 - alpha
        boundary_log = math.fsum((alpha * math.log(available_x), complement * math.log(available_y)))
        return math.log(available_tail) <= boundary_log
    if x < 0.0 or y < 0.0:
        return False
    if tail == 0.0:
        return True
    if x == 0.0 or y == 0.0:
        return False
    absolute_tail = abs(tail)
    if x == absolute_tail and y == absolute_tail:
        return True
    if alpha == 0.5 and _exact_products_equal(x, y, absolute_tail, absolute_tail):
        return True
    return _power_3d_membership_sign(x, y, tail, alpha, dual=False) > 0


def _in_power_3d_polar(
    x: float,
    y: float,
    tail: float,
    alpha: float,
    tolerance: float = 0.0,
) -> bool:
    """Test membership in the polar 3D power cone."""

    if tolerance != 0.0:
        if x > tolerance or y > tolerance:
            return False
        available_tail = max(abs(tail) - tolerance, 0.0)
        if available_tail == 0.0:
            return True
        available_x = -x + tolerance
        available_y = -y + tolerance
        if available_x <= 0.0 or available_y <= 0.0:
            return False
        complement = 1.0 - alpha
        boundary_log = math.fsum(
            (
                alpha * _log_ratio(available_x, alpha),
                complement * _log_ratio(available_y, complement),
            )
        )
        return math.log(available_tail) <= boundary_log
    if x > 0.0 or y > 0.0:
        return False
    if tail == 0.0:
        return True
    if x == 0.0 or y == 0.0:
        return False
    positive_x = -x
    positive_y = -y
    absolute_tail = abs(tail)
    if _exact_products_equal(alpha, absolute_tail, positive_x, 1.0) and _exact_products_equal(
        1.0 - alpha,
        absolute_tail,
        positive_y,
        1.0,
    ):
        return True
    if alpha == 0.5 and _exact_products_equal(
        positive_x,
        positive_y,
        absolute_tail,
        absolute_tail,
        left_multiplier=4,
    ):
        return True
    return _power_3d_membership_sign(positive_x, positive_y, tail, alpha, dual=True) > 0


def _power_3d_membership_sign(
    x: float,
    y: float,
    tail: float,
    alpha: float,
    *,
    dual: bool,
) -> int:
    """Return the sign of power-cone log slack, resolving ambiguity."""

    tail = abs(tail)
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
    for candidate_precision in (precision, 2 * precision):
        lower, upper = _power_3d_decimal_residual_interval(
            x,
            y,
            tail,
            alpha,
            dual=dual,
            precision=candidate_precision,
        )
        if lower >= 0:
            return 1
        if upper < 0:
            return -1
    # A rounded transcendental equality is not evidence of membership. Route
    # the unresolved point through the solver path instead of guessing a sign.
    return 0


def _power_3d_decimal_residual_interval(
    x: float,
    y: float,
    tail: float,
    alpha: float,
    *,
    dual: bool,
    precision: int,
) -> tuple[Decimal, Decimal]:
    """Enclose the exact log residual using directed Decimal arithmetic."""

    with localcontext() as context:
        context.prec = precision
        decimal_alpha = Decimal.from_float(alpha)
        decimal_complement = Decimal(1) - decimal_alpha
        log_x = Decimal.from_float(x).ln()
        log_y = Decimal.from_float(y).ln()
        log_tail = Decimal.from_float(tail).ln()
        log_alpha = decimal_alpha.ln() if dual else None
        log_complement = decimal_complement.ln() if dual else None
        log_bounds = {
            "x": (log_x.next_minus(), log_x.next_plus()),
            "y": (log_y.next_minus(), log_y.next_plus()),
            "tail": (log_tail.next_minus(), log_tail.next_plus()),
        }
        if dual:
            assert log_alpha is not None and log_complement is not None
            log_bounds["alpha"] = (log_alpha.next_minus(), log_alpha.next_plus())
            log_bounds["complement"] = (
                log_complement.next_minus(),
                log_complement.next_plus(),
            )

        with localcontext(context) as lower_context:
            lower_context.rounding = ROUND_FLOOR
            lower_x = log_bounds["x"][0] - log_bounds["tail"][1]
            lower_y = log_bounds["y"][0] - log_bounds["tail"][1]
            if dual:
                lower_x -= log_bounds["alpha"][1]
                lower_y -= log_bounds["complement"][1]
            lower = decimal_alpha * lower_x + decimal_complement * lower_y

        with localcontext(context) as upper_context:
            upper_context.rounding = ROUND_CEILING
            upper_x = log_bounds["x"][1] - log_bounds["tail"][0]
            upper_y = log_bounds["y"][1] - log_bounds["tail"][0]
            if dual:
                upper_x -= log_bounds["alpha"][0]
                upper_y -= log_bounds["complement"][0]
            upper = decimal_alpha * upper_x + decimal_complement * upper_y
    return lower, upper


def _exact_products_equal(
    first: float,
    second: float,
    third: float,
    fourth: float,
    *,
    left_multiplier: int = 1,
) -> bool:
    """Compare two positive float products without overflow or rounding."""

    first_numerator, first_denominator = first.as_integer_ratio()
    second_numerator, second_denominator = second.as_integer_ratio()
    third_numerator, third_denominator = third.as_integer_ratio()
    fourth_numerator, fourth_denominator = fourth.as_integer_ratio()
    return (
        left_multiplier * first_numerator * second_numerator * third_denominator * fourth_denominator
        == third_numerator * fourth_numerator * first_denominator * second_denominator
    )


def _power_3d_log_sum_uncertainty(
    logs: Sequence[float],
    terms: Sequence[float],
    residual: float,
    weights: Sequence[float],
    *,
    log_errors: Sequence[float],
) -> float:
    contributions = [math.ulp(residual)]
    for log_error, term, weight in zip(log_errors, terms, weights, strict=True):
        contributions.append(abs(weight) * log_error)
        contributions.append(2.0 * math.ulp(term))
    return math.fsum(contributions)


def _power_3d_decimal_precision(alpha: float) -> int:
    smallest_weight = min(alpha, 1.0 - alpha)
    return max(80, math.ceil(-math.log10(smallest_weight)) + 50)


def _power_3d_uses_endpoint_limit(alpha: float) -> bool:
    """Return whether the exponent is below the solver's resolution."""

    return -_power_3d_log_dual_scale(alpha) <= _ENDPOINT_RESOLUTION


def _project_power_3d_endpoint(
    vector: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Project onto the resolved endpoint limit of a 3D power cone."""

    x_value, y_value, tail = (float(entry) for entry in vector)
    if alpha <= 0.5:
        projected_y, projected_tail = _project_two_dimensional_soc(y_value, tail)
        return np.asarray([max(x_value, 0.0), projected_y, projected_tail])
    projected_x, projected_tail = _project_two_dimensional_soc(x_value, tail)
    return np.asarray([projected_x, max(y_value, 0.0), projected_tail])


def _project_two_dimensional_soc(head: float, tail: float) -> tuple[float, float]:
    magnitude = abs(tail)
    if magnitude <= head:
        return head, tail
    if magnitude <= -head:
        return 0.0, 0.0
    projected_head = 0.5 * (head + magnitude)
    return projected_head, math.copysign(projected_head, tail)
