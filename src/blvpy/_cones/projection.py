"""Batched auxiliary-solver projections for nonlinear cone diagnostics."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .exponential import _in_exponential_polar, _in_exponential_primal
from .numeric import (
    _binary_normalization_is_exact,
    _binary_normalize,
    _binary_rescale,
    _finite_norm,
    _saturated_hypot,
)
from .power import (
    _in_power_3d_polar,
    _in_power_3d_primal,
    _power_3d_dual_scale,
    _power_3d_uses_endpoint_limit,
    _project_power_3d_endpoint,
)

_SCS_EPSILON = 1e-9
_SCS_QP_EPSILON = 1e-12
_CLARABEL_EPSILON = 1e-10
_VALIDATION_FACTOR = 100.0
_INACCURATE_SOLUTION_WARNING = (
    "Solution may be inaccurate. Try another solver, adjusting the solver settings, "
    "or solve with verbose=True for more information."
)


@dataclass(slots=True)
class _ProjectionStats:
    """Counters collected by the opt-in nonlinear cone stress audit."""

    scs_batches: int = 0
    scs_blocks: int = 0
    clarabel_retries: int = 0
    clarabel_accepts: int = 0
    endpoint_limits: int = 0
    fallbacks: int = 0


_ACTIVE_STATS: ContextVar[_ProjectionStats | None] = ContextVar("blvpy_projection_stats", default=None)


@contextmanager
def _collect_projection_stats() -> Iterator[_ProjectionStats]:
    """Collect projection-path counters in the current execution context."""

    statistics = _ProjectionStats()
    token = _ACTIVE_STATS.set(statistics)
    try:
        yield statistics
    finally:
        _ACTIVE_STATS.reset(token)


@dataclass(frozen=True, slots=True)
class _ProjectionRequest:
    kind: Literal["exponential", "power_3d"]
    original: NDArray[np.float64]
    target: NDArray[np.float64]
    exponent: int
    dual: bool
    alpha: float | None = None


def _nonlinear_cone_distance(
    exponential: Sequence[NDArray[np.float64]],
    power_3d: Sequence[tuple[NDArray[np.float64], float]],
    *,
    dual: bool,
) -> float:
    """Return the combined EXP/P3D distance for one product-cone vector."""

    distance = 0.0
    requests: list[_ProjectionRequest] = []
    for vector in exponential:
        vector = np.asarray(vector, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            return float("inf")
        if _is_exponential_member(vector, dual=dual):
            continue
        request = _prepare_request("exponential", vector, dual=dual)
        if request is None:
            distance = _saturated_hypot(distance, _fallback_distance(vector))
        else:
            requests.append(request)

    for vector, alpha in power_3d:
        vector = np.asarray(vector, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            return float("inf")
        if _is_power_3d_member(vector, alpha, dual=dual):
            continue
        request = _prepare_request("power_3d", vector, dual=dual, alpha=alpha)
        if request is None:
            distance = _saturated_hypot(distance, _fallback_distance(vector))
        elif _power_3d_uses_endpoint_limit(alpha):
            projected = _project_power_3d_endpoint(request.target, alpha)
            _increment_statistic("endpoint_limits")
            endpoint_distance = _request_distance(request, projected)
            if endpoint_distance > 0.0:
                distance = _saturated_hypot(distance, endpoint_distance)
            else:
                requests.append(request)
        else:
            requests.append(request)

    if not requests:
        return distance

    primary = _solve_projection_batch(requests, solver="SCS")
    for request, projected in zip(requests, primary, strict=True):
        block_distance = None
        primary_is_valid = projected is not None and _valid_projection(request, projected, _SCS_EPSILON)
        if primary_is_valid:
            if _distance_exceeds_resolution(request, projected, _SCS_EPSILON):
                candidate_distance = _request_distance(request, projected)
                if candidate_distance > 0.0:
                    block_distance = candidate_distance
        can_refine = request.kind == "exponential" or (
            request.alpha is not None and not _power_3d_uses_endpoint_limit(request.alpha)
        )
        if block_distance is None and can_refine:
            refinement = _solve_projection_batch((request,), solver="SCS_QP")[0]
            if (
                refinement is not None
                and _valid_projection(request, refinement, _SCS_QP_EPSILON)
                and _distance_exceeds_resolution(request, refinement, _SCS_QP_EPSILON)
            ):
                candidate_distance = _request_distance(request, refinement)
                if candidate_distance > 0.0:
                    block_distance = candidate_distance
        if block_distance is None:
            retry = _solve_projection_batch((request,), solver="CLARABEL")[0]
            if retry is not None and _valid_projection(request, retry, _CLARABEL_EPSILON):
                candidate_distance = _request_distance(request, retry)
                if candidate_distance > 0.0:
                    block_distance = candidate_distance
                    _increment_statistic("clarabel_accepts")
            if block_distance is None:
                block_distance = _fallback_distance(request.original)
        distance = _saturated_hypot(distance, block_distance)
    return distance


def _prepare_request(
    kind: Literal["exponential", "power_3d"],
    vector: NDArray[np.float64],
    *,
    dual: bool,
    alpha: float | None = None,
) -> _ProjectionRequest | None:
    normalized, exponent = _binary_normalize(vector)
    if not _binary_normalization_is_exact(vector, normalized, exponent):
        return None
    return _ProjectionRequest(kind, vector, normalized, exponent, dual, alpha)


def _is_exponential_member(
    vector: NDArray[np.float64],
    *,
    dual: bool,
    tolerance: float = 0.0,
) -> bool:
    if dual:
        return _in_exponential_polar(*(-vector), tolerance)
    return _in_exponential_primal(*vector, tolerance)


def _is_power_3d_member(
    vector: NDArray[np.float64],
    alpha: float,
    *,
    dual: bool,
    tolerance: float = 0.0,
) -> bool:
    if dual:
        return _in_power_3d_polar(*(-vector), alpha, tolerance)
    return _in_power_3d_primal(*vector, alpha, tolerance)


def _solve_projection_batch(
    requests: Sequence[_ProjectionRequest],
    *,
    solver: Literal["SCS", "SCS_QP", "CLARABEL"],
) -> list[NDArray[np.float64] | None]:
    """Solve one normalized primal- or dual-cone projection problem."""

    if not requests:
        return []
    if solver == "SCS_QP":
        if len(requests) != 1:
            return [None] * len(requests)
        request = requests[0]
        if request.kind == "power_3d" and (request.alpha is None or _power_3d_uses_endpoint_limit(request.alpha)):
            return [None]

    if solver in {"SCS", "SCS_QP"}:
        _increment_statistic("scs_batches")
        _increment_statistic("scs_blocks", len(requests))
    else:
        _increment_statistic("clarabel_retries", len(requests))

    dual = requests[0].dual
    if any(request.dual != dual for request in requests[1:]):
        return [None] * len(requests)

    exponential_indices = [index for index, request in enumerate(requests) if request.kind == "exponential"]
    power_indices = [index for index, request in enumerate(requests) if request.kind == "power_3d"]
    constraints: list[cp.Constraint] = []
    differences: list[cp.Expression] = []
    variables: list[tuple[list[int], cp.Variable]] = []

    if exponential_indices:
        points = np.column_stack([requests[index].target for index in exponential_indices])
        projected = cp.Variable(points.shape)
        if dual:
            constraints.append(cp.ExpCone(-projected[1, :], -projected[0, :], math.e * projected[2, :]))
        else:
            constraints.append(cp.ExpCone(projected[0, :], projected[1, :], projected[2, :]))
        differences.append(cp.vec(projected - points, order="F"))
        variables.append((exponential_indices, projected))
    if power_indices:
        points = np.column_stack([requests[index].target for index in power_indices])
        alphas = np.asarray([requests[index].alpha for index in power_indices], dtype=np.float64)
        projected = cp.Variable(points.shape)
        tails: cp.Expression = projected[2, :]
        if dual:
            dual_scales = np.asarray([_power_3d_dual_scale(float(alpha)) for alpha in alphas])
            tails = cp.multiply(dual_scales, tails)
        constraints.append(cp.PowCone3D(projected[0, :], projected[1, :], tails, alphas))
        differences.append(cp.vec(projected - points, order="F"))
        variables.append((power_indices, projected))

    difference = differences[0] if len(differences) == 1 else cp.hstack(differences)
    objective_expression = 0.5 * cp.sum_squares(difference) if solver == "SCS_QP" else cp.norm(difference, 2)
    problem = cp.Problem(cp.Minimize(objective_expression), constraints)
    options: dict[str, object]
    if solver in {"SCS", "SCS_QP"}:
        epsilon = _SCS_QP_EPSILON if solver == "SCS_QP" else _SCS_EPSILON
        options = {
            "solver": cp.SCS,
            "eps_abs": epsilon,
            "eps_rel": epsilon,
            "max_iters": 20_000,
        }
        if solver == "SCS_QP":
            options["use_quad_obj"] = True
    else:
        options = {
            "solver": cp.CLARABEL,
            "tol_gap_abs": _CLARABEL_EPSILON,
            "tol_gap_rel": _CLARABEL_EPSILON,
            "tol_feas": _CLARABEL_EPSILON,
            "max_iter": 500,
        }
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            objective_value = problem.solve(warm_start=False, verbose=False, **options)
    except Exception:
        return [None] * len(requests)
    allowed_statuses = {cp.OPTIMAL} if solver in {"SCS", "SCS_QP"} else {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if problem.status not in allowed_statuses:
        return [None] * len(requests)
    if caught_warnings and not (
        problem.status == cp.OPTIMAL_INACCURATE
        and all(
            issubclass(warning.category, UserWarning) and str(warning.message) == _INACCURATE_SOLUTION_WARNING
            for warning in caught_warnings
        )
    ):
        return [None] * len(requests)
    if objective_value is None or not math.isfinite(float(objective_value)):
        return [None] * len(requests)

    result: list[NDArray[np.float64] | None] = [None] * len(requests)
    for indices, variable in variables:
        if variable.value is None:
            continue
        values = np.asarray(variable.value, dtype=np.float64)
        if values.shape != variable.shape:
            continue
        for column, index in enumerate(indices):
            candidate = values[:, column]
            if np.all(np.isfinite(candidate)):
                result[index] = candidate.copy()
    return result


def _valid_projection(
    request: _ProjectionRequest,
    projected: NDArray[np.float64],
    solver_epsilon: float,
) -> bool:
    target = request.target
    residual = target - projected
    target_norm = _finite_norm(target)
    projection_norm = _finite_norm(projected)
    residual_norm = _finite_norm(residual)
    tolerance = _projection_tolerance(target_norm, projection_norm, residual_norm, solver_epsilon)
    if projection_norm > target_norm + tolerance or residual_norm > target_norm + tolerance:
        return False
    if request.kind == "exponential":
        projection_member = _is_exponential_member(projected, dual=request.dual, tolerance=tolerance)
        opposite_member = _is_exponential_member(-residual, dual=not request.dual, tolerance=tolerance)
    else:
        assert request.alpha is not None
        projection_member = _is_power_3d_member(
            projected,
            request.alpha,
            dual=request.dual,
            tolerance=tolerance,
        )
        opposite_member = _is_power_3d_member(
            -residual,
            request.alpha,
            dual=not request.dual,
            tolerance=tolerance,
        )
    pairing = abs(float(np.dot(projected, residual)))
    return projection_member and opposite_member and math.isfinite(pairing) and pairing <= tolerance


def _request_distance(
    request: _ProjectionRequest,
    projected: NDArray[np.float64],
) -> float:
    normalized_distance = _normalized_request_distance(request, projected)
    return _binary_rescale(normalized_distance, request.exponent)


def _distance_exceeds_resolution(
    request: _ProjectionRequest,
    projected: NDArray[np.float64],
    solver_epsilon: float,
) -> bool:
    residual = request.target - projected
    tolerance = _projection_tolerance(
        _finite_norm(request.target),
        _finite_norm(projected),
        _finite_norm(residual),
        solver_epsilon,
    )
    return _normalized_request_distance(request, projected) > tolerance


def _normalized_request_distance(
    request: _ProjectionRequest,
    projected: NDArray[np.float64],
) -> float:
    return _finite_norm(request.target - projected)


def _projection_tolerance(
    target_norm: float,
    projection_norm: float,
    residual_norm: float,
    solver_epsilon: float,
) -> float:
    return _VALIDATION_FACTOR * solver_epsilon * (1.0 + target_norm + projection_norm + residual_norm)


def _fallback_distance(vector: NDArray[np.float64]) -> float:
    _increment_statistic("fallbacks")
    return _finite_norm(vector)


def _increment_statistic(name: str, amount: int = 1) -> None:
    statistics = _ACTIVE_STATS.get()
    if statistics is not None:
        setattr(statistics, name, getattr(statistics, name) + amount)
