"""Internal construction of public gap diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from .backends import solve_conic
from .errors import SolveError, SolverUnavailableError
from .result import BilevelResult, GapDiagnostics

if TYPE_CHECKING:
    from .problem import BilevelProblem


def _gap_diagnostics(
    model: BilevelProblem,
    result: BilevelResult,
    *,
    solver: str = cp.CLARABEL,
    solver_options: Mapping[str, Any] | None = None,
    solver_verbose: bool = False,
) -> GapDiagnostics:
    """Build complete diagnostics for one result without changing model state."""

    if not isinstance(result, BilevelResult):
        raise TypeError("result must be a BilevelResult.")
    if not result.succeeded and result.status.lower() != "continuation_failed":
        raise ValueError("Gap diagnostics require a successful or continuation_failed BilevelResult.")
    if solver is None:
        raise ValueError("solver must name a CVXPY conic backend; None is not supported.")
    if not isinstance(solver_verbose, (bool, np.bool_)):
        raise ValueError("solver_verbose must be boolean.")
    solve_options = dict(solver_options) if solver_options is not None else {}

    source_values = _source_value_snapshots(model, result.variable_values)
    canonical = model.canonicalize()
    primal = _result_vector(result.canonical_primal, "canonical_primal", canonical.canonical_size)
    slack = _result_vector(result.slack, "slack", canonical.constraint_size)
    dual = _result_vector(result.dual, "dual", canonical.constraint_size)
    parameter_values = {parameter: source_values[variable] for parameter, variable in model._parameter_links.items()}
    data = canonical.apply_numeric(parameter_values)
    primal_residual = np.asarray(data.A @ primal + slack - data.b, dtype=float)
    dual_residual = np.asarray(data.A.T @ dual + data.c, dtype=float)

    affected_variables = model.source_variables
    affected_parameters = _lower_parameters(model)
    variable_states = {variable: _snapshot_leaf(variable) for variable in affected_variables}
    parameter_states = {parameter: _snapshot_leaf(parameter) for parameter in affected_parameters}
    try:
        for variable, value in source_values.items():
            _save_dense_value(variable, value)
        _assign_reference_parameters(model, canonical.fixed_parameter_values, source_values)

        returned_value = _source_objective_value(model)
        reference_value = _solve_reference_lower(
            model,
            solver,
            solve_options,
            bool(solver_verbose),
        )
        if isinstance(model.lower_problem.objective, cp.Maximize):
            source_gap = reference_value - returned_value
        else:
            source_gap = returned_value - reference_value
    finally:
        for variable, state in variable_states.items():
            _restore_leaf(variable, state)
        for parameter, state in parameter_states.items():
            _restore_leaf(parameter, state)

    return _from_canonical(
        c=data.c,
        b=data.b,
        primal=primal,
        dual=dual,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        complementarity=float(slack @ dual),
        source_gap=source_gap,
    )


def _from_canonical(
    *,
    c: ArrayLike,
    b: ArrayLike,
    primal: ArrayLike,
    dual: ArrayLike,
    primal_residual: ArrayLike,
    dual_residual: ArrayLike,
    complementarity: float,
    source_gap: float | None = None,
) -> GapDiagnostics:
    """Construct the inexact-gap identity from compatible canonical vectors."""

    c_vector = _finite_vector(c, "c")
    b_vector = _finite_vector(b, "b")
    primal_vector = _finite_vector(primal, "primal")
    dual_vector = _finite_vector(dual, "dual")
    primal_residual_vector = _finite_vector(primal_residual, "primal_residual")
    dual_residual_vector = _finite_vector(dual_residual, "dual_residual")
    _require_same_size(c_vector, primal_vector, "c", "primal")
    _require_same_size(b_vector, dual_vector, "b", "dual")
    _require_same_size(primal_residual_vector, dual_vector, "primal_residual", "dual")
    _require_same_size(dual_residual_vector, primal_vector, "dual_residual", "primal")
    return GapDiagnostics(
        primal_objective=float(c_vector @ primal_vector),
        dual_objective=float(-(b_vector @ dual_vector)),
        complementarity=complementarity,
        dual_residual_term=float(primal_vector @ dual_residual_vector),
        primal_residual_term=float(dual_vector @ primal_residual_vector),
        source_gap=source_gap,
    )


def _source_value_snapshots(
    model: BilevelProblem,
    values: Mapping[Any, ArrayLike],
) -> dict[cp.Variable, NDArray[np.float64]]:
    expected = {variable.id: variable for variable in model.source_variables}
    actual: dict[int, cp.Variable] = {}
    for key in values:
        if not isinstance(key, cp.Variable):
            raise ValueError("result.variable_values contains a non-variable key.")
        actual[key.id] = key
    if set(actual) != set(expected) or any(actual[key] is not expected[key] for key in actual):
        raise ValueError("result.variable_values has incompatible source variables for this BilevelProblem.")

    snapshots: dict[cp.Variable, NDArray[np.float64]] = {}
    for variable in model.source_variables:
        array = _finite_array(values[variable], f"variable_values[{variable.name()!r}]")
        if array.shape != variable.shape:
            raise ValueError(
                f"variable_values[{variable.name()!r}] has shape {array.shape}; expected {variable.shape}."
            )
        snapshots[variable] = array
    return snapshots


def _result_vector(value: ArrayLike | None, name: str, expected_size: int) -> NDArray[np.float64]:
    if value is None:
        raise ValueError(f"result.{name} is missing.")
    vector = _finite_vector(value, name)
    if vector.size != expected_size:
        raise ValueError(f"result.{name} has {vector.size} entries; expected {expected_size}.")
    return vector


def _lower_parameters(model: BilevelProblem) -> tuple[cp.Parameter, ...]:
    parameters: dict[int, cp.Parameter] = {
        parameter.id: parameter for parameter in model._cvxpy_lower_problem.parameters()
    }
    for parameter in model.lower_problem.objective.expr.parameters():
        parameters.setdefault(parameter.id, parameter)
    for constraint in model.lower_problem.constraints:
        for parameter in constraint.parameters():
            parameters.setdefault(parameter.id, parameter)
    return tuple(parameters.values())


def _assign_reference_parameters(
    model: BilevelProblem,
    fixed_values: Mapping[int, ArrayLike],
    source_values: Mapping[cp.Variable, NDArray[np.float64]],
) -> None:
    linked_ids = {parameter.id for parameter in model._parameter_links}
    for parameter, variable in model._parameter_links.items():
        _save_dense_value(parameter, source_values[variable])
    for parameter in _lower_parameters(model):
        if parameter.id in linked_ids:
            continue
        try:
            value = fixed_values[parameter.id]
        except KeyError as error:
            raise ValueError(
                f"No canonicalization-time value is available for fixed lower parameter {parameter.name()!r}."
            ) from error
        _save_dense_value(parameter, value)


def _source_objective_value(model: BilevelProblem) -> float:
    value = model.lower_problem.objective.expr.value
    if value is None:
        raise ValueError("The returned source lower objective could not be evaluated.")
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array):
        raise ValueError("The returned source lower objective is not a real scalar.")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError("The returned source lower objective is not finite.")
    return result


def _solve_reference_lower(
    model: BilevelProblem,
    solver: str,
    solver_options: Mapping[str, Any],
    solver_verbose: bool,
) -> float:
    generated = model._cvxpy_lower_problem
    objective = generated.objective.tree_copy()
    constraints = [constraint.tree_copy() for constraint in generated.constraints]
    reference = cp.Problem(objective, constraints)
    try:
        solve_conic(reference, solver, solver_options, solver_verbose)
    except SolverUnavailableError:
        raise
    except cp.SolverError as error:
        raise SolveError(f"The fixed-upper lower reference solve failed: {error}") from error
    if reference.status not in cp.settings.SOLUTION_PRESENT:
        raise SolveError(f"The fixed-upper lower reference problem returned status {reference.status!r}.")
    if reference.value is None:
        raise SolveError("The fixed-upper lower reference solve returned no objective value.")
    value = np.asarray(reference.value)
    if value.shape != () or np.iscomplexobj(value) or not np.isfinite(value):
        raise SolveError("The fixed-upper lower reference solve returned an invalid objective value.")
    return float(value)


@dataclass(frozen=True, slots=True)
class _LeafState:
    value: Any
    sparse_path: bool


def _snapshot_leaf(leaf: cp.Variable | cp.Parameter) -> _LeafState:
    if leaf.sparse_idx is not None:
        value = leaf.value_sparse
        data = None if value is None else np.array(value.data, copy=True)
        return _LeafState(data, True)
    value = leaf.value
    if value is None:
        snapshot = None
    elif sp.issparse(value):
        snapshot = value.copy()
    else:
        snapshot = np.array(value, copy=True)
    return _LeafState(snapshot, False)


def _restore_leaf(leaf: cp.Variable | cp.Parameter, state: _LeafState) -> None:
    if state.value is None:
        leaf.save_value(None)
    elif state.sparse_path:
        leaf.save_value(np.array(state.value, copy=True), sparse_path=True)
    elif sp.issparse(state.value):
        leaf.save_value(state.value.copy())
    else:
        leaf.save_value(np.array(state.value, copy=True))


def _save_dense_value(leaf: cp.Variable | cp.Parameter, value: ArrayLike) -> None:
    array = np.array(value, dtype=float, copy=True)
    if leaf.sparse_idx is None:
        leaf.save_value(array)
    else:
        leaf.save_value(array[leaf.sparse_idx], sparse_path=True)


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.array(array, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _finite_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    return _finite_array(value, name).reshape(-1, order="F")


def _require_same_size(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    left_name: str,
    right_name: str,
) -> None:
    if left.size != right.size:
        raise ValueError(f"{left_name} has {left.size} entries but {right_name} has {right.size}.")
