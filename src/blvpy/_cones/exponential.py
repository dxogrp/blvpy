"""Membership predicates for the exponential cone and its polar."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

from .numeric import _log_ratio, _log_ratio_roundoff


def _decimal_membership_precision(*values: float) -> int:
    """Return enough precision to preserve the scale of input floats."""

    exponents = [abs(math.frexp(value)[1]) for value in values if value != 0.0]
    return max(100, math.ceil(max(exponents, default=0) * math.log10(2.0)) + 100)


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
    """Test membership in the polar EXP cone in logarithmic form."""

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
    """Return the sign of EXP log slack, resolving rounded boundaries."""

    if polar and positive_third == first:
        return (first > second) - (first < second)
    if not polar and positive_third == second:
        return (0.0 > first) - (0.0 < first)
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
        context.prec = _decimal_membership_precision(first, second, positive_third)
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
