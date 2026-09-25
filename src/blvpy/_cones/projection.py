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
    _power_3d_uses_endpoint_limit,
    _project_power_3d_endpoint,
)

_SCS_EPSILON = 1e-9
_CLARABEL_EPSILON = 1e-10
_VALIDATION_FACTOR = 100.0


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
            distance = _saturated_hypot(distance, _request_distance(request, projected))
            _increment_statistic("endpoint_limits")
        else:
            requests.append(request)

    if not requests:
        return distance

    primary = _solve_projection_batch(requests, solver="SCS")
    for request, projected in zip(requests, primary, strict=True):
        if projected is not None and _valid_projection(request, projected, _SCS_EPSILON):
            block_distance = _request_distance(request, projected, solver_epsilon=_SCS_EPSILON)
        else:
            retry = _solve_projection_batch((request,), solver="CLARABEL")[0]
            if retry is not None and _valid_projection(request, retry, _CLARABEL_EPSILON):
                block_distance = _request_distance(request, retry, solver_epsilon=_CLARABEL_EPSILON)
                _increment_statistic("clarabel_accepts")
            else:
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
    target = -normalized if dual else normalized
    return _ProjectionRequest(kind, vector, target, exponent, dual, alpha)


def _is_exponential_member(vector: NDArray[np.float64], *, dual: bool) -> bool:
    if dual:
        return _in_exponential_polar(*(-vector))
    return _in_exponential_primal(*vector)


def _is_power_3d_member(vector: NDArray[np.float64], alpha: float, *, dual: bool) -> bool:
    if dual:
        return _in_power_3d_polar(*(-vector), alpha)
    return _in_power_3d_primal(*vector, alpha)


def _solve_projection_batch(
    requests: Sequence[_ProjectionRequest],
    *,
    solver: Literal["SCS", "CLARABEL"],
) -> list[NDArray[np.float64] | None]:
    """Solve one normalized primal-cone projection problem."""

    if not requests:
        return []
    if solver == "SCS":
        _increment_statistic("scs_batches")
        _increment_statistic("scs_blocks", len(requests))
    else:
        _increment_statistic("clarabel_retries", len(requests))

    exponential_indices = [index for index, request in enumerate(requests) if request.kind == "exponential"]
    power_indices = [index for index, request in enumerate(requests) if request.kind == "power_3d"]
    constraints: list[cp.Constraint] = []
    differences: list[cp.Expression] = []
    variables: list[tuple[list[int], cp.Variable]] = []

    if exponential_indices:
        points = np.column_stack([requests[index].target for index in exponential_indices])
        projected = cp.Variable(points.shape)
        constraints.append(cp.ExpCone(projected[0, :], projected[1, :], projected[2, :]))
        differences.append(cp.vec(projected - points, order="F"))
        variables.append((exponential_indices, projected))
    if power_indices:
        points = np.column_stack([requests[index].target for index in power_indices])
        alphas = np.asarray([requests[index].alpha for index in power_indices], dtype=np.float64)
        projected = cp.Variable(points.shape)
        constraints.append(cp.PowCone3D(projected[0, :], projected[1, :], projected[2, :], alphas))
        differences.append(cp.vec(projected - points, order="F"))
        variables.append((power_indices, projected))

    difference = differences[0] if len(differences) == 1 else cp.hstack(differences)
    problem = cp.Problem(cp.Minimize(cp.norm(difference, 2)), constraints)
    options: dict[str, object]
    if solver == "SCS":
        options = {
            "solver": cp.SCS,
            "eps_abs": _SCS_EPSILON,
            "eps_rel": _SCS_EPSILON,
            "max_iters": 20_000,
        }
    else:
        options = {
            "solver": cp.CLARABEL,
            "tol_gap_abs": _CLARABEL_EPSILON,
            "tol_gap_rel": _CLARABEL_EPSILON,
            "tol_feas": _CLARABEL_EPSILON,
            "max_iter": 500,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            objective = problem.solve(warm_start=False, verbose=False, **options)
    except Exception:
        return [None] * len(requests)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return [None] * len(requests)
    if objective is None or not math.isfinite(float(objective)):
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
        primal_member = _in_exponential_primal(*projected, tolerance)
        polar_member = _in_exponential_polar(*residual, tolerance)
    else:
        assert request.alpha is not None
        primal_member = _in_power_3d_primal(*projected, request.alpha, tolerance)
        polar_member = _in_power_3d_polar(*residual, request.alpha, tolerance)
    pairing = abs(float(np.dot(projected, residual)))
    return primal_member and polar_member and math.isfinite(pairing) and pairing <= tolerance


def _request_distance(
    request: _ProjectionRequest,
    projected: NDArray[np.float64],
    *,
    solver_epsilon: float | None = None,
) -> float:
    normalized_distance = _finite_norm(projected) if request.dual else _finite_norm(request.target - projected)
    if solver_epsilon is not None:
        residual = request.target - projected
        tolerance = _projection_tolerance(
            _finite_norm(request.target),
            _finite_norm(projected),
            _finite_norm(residual),
            solver_epsilon,
        )
        if normalized_distance <= tolerance:
            return 0.0
    return _binary_rescale(normalized_distance, request.exponent)


def _projection_tolerance(
    target_norm: float,
    projection_norm: float,
    residual_norm: float,
    solver_epsilon: float,
) -> float:
    return _VALIDATION_FACTOR * solver_epsilon * (
        1.0 + target_norm + projection_norm + residual_norm
    )


def _fallback_distance(vector: NDArray[np.float64]) -> float:
    _increment_statistic("fallbacks")
    return _finite_norm(vector)


def _increment_statistic(name: str, amount: int = 1) -> None:
    statistics = _ACTIVE_STATS.get()
    if statistics is not None:
        setattr(statistics, name, getattr(statistics, name) + amount)
