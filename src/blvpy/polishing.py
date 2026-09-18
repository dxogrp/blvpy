"""Fixed-upper lower polishing and candidate assessment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from .continuation import compute_residuals
from .diagnostics import (
    _assign_reference_parameters,
    _lower_parameters,
    _restore_leaf,
    _save_dense_value,
    _snapshot_leaf,
    _source_value_snapshots,
)
from .errors import SolveError, SolverUnavailableError
from .fixed_lower import FixedLowerSolveError, solve_fixed_lower
from .progress import ProgressReporter
from .result import BilevelResult, PolishResult

if TYPE_CHECKING:
    from .problem import BilevelProblem


@dataclass(frozen=True, slots=True)
class _PolishSettings:
    """Numerical and reporting settings for one polishing solve."""

    solver: str = cp.CLARABEL
    solver_options: Mapping[str, Any] | None = None
    verbose: bool = True
    solver_verbose: bool = False


def polish_bilevel(
    model: BilevelProblem,
    result: BilevelResult,
    settings: _PolishSettings,
) -> PolishResult:
    """Validate, polish, and report one bilevel result."""

    progress_verbose = _boolean(settings.verbose, "verbose")
    reporter = ProgressReporter(enabled=progress_verbose)
    started_at = perf_counter()
    try:
        solver_verbose = _boolean(settings.solver_verbose, "solver_verbose")
        polished = _polish_bilevel(model, result, settings, solver_verbose)
    except Exception as error:
        reporter.failure(error, elapsed=perf_counter() - started_at)
        raise
    reporter.polishing(polished)
    return polished


def _polish_bilevel(
    model: BilevelProblem,
    result: BilevelResult,
    settings: _PolishSettings,
    solver_verbose: bool,
) -> PolishResult:
    if not isinstance(result, BilevelResult):
        raise TypeError("result must be a BilevelResult.")
    if not result.succeeded and result.status.lower() != "continuation_failed":
        raise ValueError("Polishing requires a successful or continuation_failed BilevelResult.")
    if result._problem_token is not None and result._problem_token is not model._result_token:
        raise ValueError("result was produced by a different BilevelProblem.")
    if settings.solver is None:
        raise ValueError("solver must name a CVXPY conic backend; None is not supported.")
    solve_options = dict(settings.solver_options) if settings.solver_options is not None else {}

    model.validate()
    source_values = _source_value_snapshots(model, result.variable_values)
    canonical = model.canonicalize()
    lifted = model._lifted_problem
    affected_variables = (*model.source_variables, lifted.primal, lifted.slack, lifted.dual)
    affected_parameters = _lower_parameters(model)
    variable_states = {variable: _snapshot_leaf(variable) for variable in affected_variables}
    parameter_states = {parameter: _snapshot_leaf(parameter) for parameter in affected_parameters}
    try:
        for variable, value in source_values.items():
            _save_dense_value(variable, value)
        _assign_reference_parameters(model, canonical.fixed_parameter_values, source_values)
        original_objective = _upper_objective_value(model, "original")

        try:
            solution = solve_fixed_lower(
                model,
                settings.solver,
                solve_options,
                solver_verbose,
            )
        except SolverUnavailableError:
            raise
        except (cp.SolverError, FixedLowerSolveError) as error:
            raise SolveError(f"The fixed-upper lower polishing solve failed: {error}") from error

        _assign_polished_lower_values(model, solution.source_values)
        _save_dense_value(lifted.primal, solution.primal)
        _save_dense_value(lifted.slack, solution.slack)
        _save_dense_value(lifted.dual, solution.dual)

        try:
            residuals = compute_residuals(model, epsilon=0.0)
        except Exception as error:
            raise SolveError(f"Could not evaluate polished feasibility: {error}") from error
        polished_objective = _upper_objective_value(model, "polished")
        candidate_values = _complete_candidate_values(model)
        return PolishResult(
            variable_values=candidate_values,
            feasible=residuals.is_feasible(result._feasibility_tolerance),
            objective=polished_objective,
            objective_improvement_ratio=_objective_improvement_ratio(
                original_objective,
                polished_objective,
                maximize=isinstance(model.upper_objective, cp.Maximize),
            ),
        )
    finally:
        for variable, state in variable_states.items():
            _restore_leaf(variable, state)
        for parameter, state in parameter_states.items():
            _restore_leaf(parameter, state)


def _upper_objective_value(model: BilevelProblem, point: str) -> float:
    value = model.upper_objective.value
    if value is None:
        raise ValueError(f"The upper objective at the {point} point could not be evaluated.")
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array):
        raise ValueError(f"The upper objective at the {point} point is not a real scalar.")
    objective = float(array)
    if not np.isfinite(objective):
        raise ValueError(f"The upper objective at the {point} point is not finite.")
    return objective


def _assign_polished_lower_values(
    model: BilevelProblem,
    source_values: Mapping[int, Any],
) -> None:
    lower_by_id = {variable.id: variable for variable in model._cvxpy_lower_problem.variables()}
    if set(source_values) != set(lower_by_id):
        raise SolveError("The fixed-upper lower solve returned incomplete recovered variable values.")
    for variable_id, variable in lower_by_id.items():
        value = np.asarray(source_values[variable_id])
        if np.iscomplexobj(value):
            raise SolveError("The fixed-upper lower solve returned complex recovered variable values.")
        try:
            value = np.array(value, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise SolveError("The fixed-upper lower solve returned invalid recovered variable values.") from error
        if value.shape != variable.shape:
            raise SolveError(
                f"The fixed-upper lower solve recovered shape {value.shape} for {variable.name()!r}; "
                f"expected {variable.shape}."
            )
        if not np.isfinite(value).all():
            raise SolveError("The fixed-upper lower solve returned nonfinite recovered variable values.")
        _save_dense_value(variable, value)


def _complete_candidate_values(model: BilevelProblem) -> dict[cp.Variable, np.ndarray]:
    values: dict[cp.Variable, np.ndarray] = {}
    for variable in model.source_variables:
        if variable.value is None:
            raise SolveError(f"The polished candidate has no value for source variable {variable.name()!r}.")
        values[variable] = np.asarray(variable.value, dtype=float)
    return values


def _objective_improvement_ratio(
    original: float,
    polished: float,
    *,
    maximize: bool,
) -> float | None:
    """Return relative objective improvement, or None for a zero baseline."""

    original = float(original)
    polished = float(polished)
    if original == 0.0:
        return None
    improvement = polished - original if maximize else original - polished
    return improvement / abs(original)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean.")
    return bool(value)


__all__: list[str] = []
