"""Constraint relaxation helpers for feasibility restoration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy as cp

from .._cones.power import _power_3d_dual_scale
from ..errors import SolveError

if TYPE_CHECKING:
    from ..problem import BilevelProblem


def _relaxed_cone_constraints(
    model: BilevelProblem,
    radius: cp.Expression,
) -> tuple[cp.Constraint, ...]:
    lifted = model._lifted_problem
    layout = model.canonicalize().cone_layout
    slack, dual = lifted.slack, lifted.dual
    constraints: list[cp.Constraint] = []
    if layout.zero:
        constraints.extend([slack[layout.zero_slice] <= radius, -slack[layout.zero_slice] <= radius])
    if layout.nonnegative:
        constraints.extend([slack[layout.nonnegative_slice] >= -radius, dual[layout.nonnegative_slice] >= -radius])
    for block in layout.second_order_slices:
        constraints.extend(
            [
                cp.norm(slack[block.start + 1 : block.stop], 2) <= slack[block.start] + radius,
                cp.norm(dual[block.start + 1 : block.stop], 2) <= dual[block.start] + radius,
            ]
        )
    for block in layout.exponential_slices:
        primal_x, primal_y, primal_z = slack[block.start : block.stop]
        dual_u, dual_v, dual_w = dual[block.start : block.stop]
        constraints.extend(
            [
                primal_y + radius >= 0,
                primal_z + radius >= 0,
                cp.rel_entr(primal_y + radius, primal_z + radius) <= -primal_x + radius,
                -dual_u + radius >= 0,
                dual_w + radius >= 0,
                cp.rel_entr(-dual_u + radius, dual_w + radius) <= dual_v - dual_u + 2.0 * radius,
            ]
        )
    for block, alpha in zip(layout.power_3d_slices, layout.power_3d, strict=True):
        slack_heads = slack[block.start : block.start + 2] + radius
        dual_heads = dual[block.start : block.start + 2] + radius
        weights = (alpha, 1.0 - alpha)
        dual_scale = _power_3d_dual_scale(alpha)
        constraints.extend(
            [
                slack_heads >= 0,
                cp.abs(slack[block.start + 2]) <= cp.geo_mean(slack_heads, p=weights, approx=False) + radius,
                dual_heads >= 0,
                dual_scale * cp.abs(dual[block.start + 2])
                <= cp.geo_mean(dual_heads, p=weights, approx=False) + dual_scale * radius,
            ]
        )
    return tuple(constraints)


def _relax_constraint(
    constraint: cp.Constraint,
    radius: cp.Expression,
) -> tuple[cp.Constraint, ...]:
    if isinstance(constraint, cp.constraints.zero.Equality):
        return constraint.expr <= radius, -constraint.expr <= radius
    if isinstance(constraint, cp.constraints.nonpos.Inequality):
        return (constraint.expr <= radius,)
    raise SolveError(f"Cannot construct feasibility restoration for {type(constraint).__name__}.")
