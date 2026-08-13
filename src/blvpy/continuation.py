"""Initialization, restoration, and epsilon-gap continuation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import InitializationError, SolveError, SolverUnavailableError
from .result import BilevelResult, IterationRecord, Residuals, StartRecord

if TYPE_CHECKING:
    from .problem import BilevelProblem, LiftedProblem

_ACCEPTABLE_NLP_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    objective: float
    state: Mapping[int, NDArray[np.float64]]
    residuals: Residuals


@dataclass(frozen=True, slots=True)
class SolveSettings:
    """Numerical settings for multistart gap continuation."""

    epsilon_initial: float = 1e-1
    epsilon_target: float = 1e-6
    contraction: float = 0.1
    starts: int = 10
    feasibility_tolerance: float = 1e-7
    seed: int | np.random.Generator | None = None
    solver: str = cp.IPOPT
    conic_solver: str = cp.CLARABEL
    solver_options: Mapping[str, Any] | None = None
    conic_solver_options: Mapping[str, Any] | None = None
    restoration: bool = True
    max_retries: int = 8
    verbose: bool = False


def solve_bilevel(model: BilevelProblem, settings: SolveSettings) -> BilevelResult:
    """Internal implementation of :meth:`BilevelProblem.solve`."""

    epsilon_initial = settings.epsilon_initial
    epsilon_target = settings.epsilon_target
    contraction = settings.contraction
    starts = settings.starts
    feasibility_tolerance = settings.feasibility_tolerance
    seed = settings.seed
    solver = settings.solver
    conic_solver = settings.conic_solver
    solver_options = settings.solver_options
    conic_solver_options = settings.conic_solver_options
    restoration = settings.restoration
    max_retries = settings.max_retries
    verbose = settings.verbose

    epsilon_initial = _finite_nonnegative(epsilon_initial, "epsilon_initial")
    epsilon_target = _finite_nonnegative(epsilon_target, "epsilon_target")
    if epsilon_initial <= 0 or epsilon_target <= 0:
        raise ValueError("epsilon_initial and epsilon_target must be positive.")
    if epsilon_target > epsilon_initial:
        raise ValueError("epsilon_target cannot exceed epsilon_initial.")
    contraction = _finite_nonnegative(contraction, "contraction")
    if not 0 < contraction < 1:
        raise ValueError("contraction must lie strictly between zero and one.")
    if isinstance(starts, bool) or not isinstance(starts, (int, np.integer)) or starts < 1:
        raise ValueError("starts must be a positive integer.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, (int, np.integer)) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer.")
    feasibility_tolerance = _finite_nonnegative(
        feasibility_tolerance, "feasibility_tolerance"
    )
    solver_options = dict(solver_options or {})
    conic_solver_options = dict(conic_solver_options or {})

    model.validate()
    lifted = model.lifted_problem
    canonical = model.canonicalize()
    _require_solver(solver, nonlinear=True)
    _require_solver(conic_solver, nonlinear=False)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    samples = sample_upper_starts(model, int(starts), rng)

    start_records: list[StartRecord] = []
    candidates: list[_Candidate] = []
    for index, sample in enumerate(samples):
        try:
            _assign_values(sample)
            _initialize_lower(model, conic_solver, conic_solver_options, verbose)
            lifted.epsilon.value = epsilon_initial
            initial_residuals = compute_residuals(model, epsilon_initial)
            if restoration and not initial_residuals.is_feasible(feasibility_tolerance):
                _restore_feasibility(
                    model,
                    epsilon_initial,
                    solver,
                    solver_options,
                    verbose,
                    feasibility_tolerance,
                )
            _compile_probe(lifted, use_hessian=_uses_exact_hessian(solver_options))
            record = _solve_one(
                model,
                epsilon_initial,
                solver,
                solver_options,
                verbose,
            )
            if not _record_is_acceptable(record, feasibility_tolerance):
                message = record.message
                if record.status in _ACCEPTABLE_NLP_STATUSES:
                    message = (
                        "Independent residual check failed: "
                        f"max violation {record.residuals.max_violation:.3g}."
                    )
                start_records.append(
                    StartRecord(
                        index,
                        "residual_check_failed"
                        if record.status in _ACCEPTABLE_NLP_STATUSES
                        else record.status,
                        record.objective,
                        record.residuals,
                        message,
                    )
                )
                continue
            state = _snapshot_state(lifted.problem)
            objective = float(record.objective) if record.objective is not None else float("inf")
            candidate = _Candidate(index, objective, state, record.residuals)
            candidates.append(candidate)
            start_records.append(
                StartRecord(index, record.status, record.objective, record.residuals)
            )
        except Exception as error:
            start_records.append(
                StartRecord(index, "failed", message=f"{type(error).__name__}: {error}")
            )

    if not candidates:
        details = "; ".join(
            f"start {record.index}: {record.message or record.status}" for record in start_records
        )
        raise InitializationError(
            "No multistart initialization produced a lifted solution. " + details
        )

    candidates.sort(key=lambda candidate: candidate.objective)
    best = candidates[0]
    _restore_state(lifted.problem, best.state)
    initial_record = IterationRecord(
        epsilon=epsilon_initial,
        status=start_records[best.index].status,
        objective=best.objective,
        residuals=best.residuals,
        solver_name=str(solver),
    )
    iterations: list[IterationRecord] = [initial_record]
    last_successful_epsilon = epsilon_initial
    last_successful_state = dict(best.state)

    targets = list(_epsilon_schedule(epsilon_initial, epsilon_target, contraction))[1:]
    scheduled_targets = set(targets)
    target_index = 0
    retries = 0
    alternatives = candidates[1:]
    while target_index < len(targets):
        epsilon = targets[target_index]
        _restore_state(lifted.problem, last_successful_state)
        record = _solve_one(model, epsilon, solver, solver_options, verbose)
        iterations.append(record)
        if _record_is_acceptable(record, feasibility_tolerance):
            last_successful_epsilon = epsilon
            last_successful_state = _snapshot_state(lifted.problem)
            target_index += 1
            if epsilon in scheduled_targets:
                retries = 0
            continue

        recovered = False
        for alternative in alternatives:
            _restore_state(lifted.problem, alternative.state)
            retry_record = _solve_one(model, epsilon, solver, solver_options, verbose)
            iterations.append(retry_record)
            if _record_is_acceptable(retry_record, feasibility_tolerance):
                last_successful_epsilon = epsilon
                last_successful_state = _snapshot_state(lifted.problem)
                target_index += 1
                if epsilon in scheduled_targets:
                    retries = 0
                recovered = True
                break
        if recovered:
            continue
        if retries >= max_retries:
            _restore_state(lifted.problem, last_successful_state)
            lifted.epsilon.value = last_successful_epsilon
            return _result(
                model,
                "continuation_failed",
                iterations,
                start_records,
                message=f"Could not reduce epsilon below {last_successful_epsilon:.6g}.",
                final_record=_restored_record(
                    model, last_successful_epsilon, solver, feasibility_tolerance
                ),
            )
        intermediate = sqrt(last_successful_epsilon * epsilon)
        if not epsilon < intermediate < last_successful_epsilon:
            _restore_state(lifted.problem, last_successful_state)
            lifted.epsilon.value = last_successful_epsilon
            return _result(
                model,
                "continuation_failed",
                iterations,
                start_records,
                message="Continuation could not insert a distinct intermediate epsilon.",
                final_record=_restored_record(
                    model, last_successful_epsilon, solver, feasibility_tolerance
                ),
            )
        targets.insert(target_index, intermediate)
        retries += 1

    _restore_state(lifted.problem, last_successful_state)
    lifted.epsilon.value = last_successful_epsilon
    try:
        final = _restored_record(model, last_successful_epsilon, solver, feasibility_tolerance)
    except Exception:
        final = next(
            record
            for record in reversed(iterations)
            if _record_is_acceptable(record, feasibility_tolerance)
        )
    status = final.status
    message = None
    if not final.residuals.is_feasible(
        feasibility_tolerance,
        gap_tolerance=feasibility_tolerance,
    ):
        status = "residual_check_failed"
        message = (
            "The NLP solver returned a point, but one or more independently "
            "computed residuals exceed the requested feasibility tolerance."
        )
    return _result(
        model,
        status,
        iterations,
        start_records,
        message=message,
        final_record=final,
    )


def sample_upper_starts(
    model: BilevelProblem,
    starts: int,
    rng: np.random.Generator,
) -> tuple[dict[cp.Variable, NDArray[np.float64]], ...]:
    """Generate reproducible upper starts from values and finite bounds."""

    specifications: list[
        tuple[
            cp.Variable,
            NDArray[np.float64] | None,
            NDArray[np.float64],
            NDArray[np.float64],
            bool,
        ]
    ] = []
    for variable in model.upper_variables:
        explicit = _numeric_value(variable.value, variable.shape) if variable.value is not None else None
        lower, upper, fully_bounded = _sampling_bounds(variable)
        if explicit is None and not fully_bounded:
            raise InitializationError(
                f"Upper variable {variable.name()!r} has an unbounded component and no "
                "initial value or finite sample_bounds."
            )
        specifications.append((variable, explicit, lower, upper, fully_bounded))

    samples: list[dict[cp.Variable, NDArray[np.float64]]] = []
    for index in range(starts):
        sample: dict[cp.Variable, NDArray[np.float64]] = {}
        for variable, explicit, lower, upper, fully_bounded in specifications:
            if index == 0 and explicit is not None:
                value = explicit
            elif fully_bounded:
                value = rng.uniform(lower, upper)
            elif explicit is not None:
                value = explicit
            else:  # guarded above
                raise AssertionError("Missing upper-variable initialization.")
            sample[variable] = np.asarray(value, dtype=float).reshape(variable.shape)
        samples.append(sample)
    return tuple(samples)


def compute_residuals(model: BilevelProblem, epsilon: float | None = None) -> Residuals:
    """Independently compute all reported lifted residuals."""

    lifted = model.lifted_problem
    if epsilon is None:
        epsilon = float(lifted.epsilon.value)
    values = {
        parameter: variable.value for parameter, variable in model.parameter_map.items()
    }
    if any(value is None for value in values.values()):
        raise InitializationError("Mapped upper variables do not all have numeric values.")
    data = model.canonicalize().apply_numeric(values)
    primal = _required_vector(lifted.primal.value, "canonical primal")
    slack = _required_vector(lifted.slack.value, "canonical slack")
    dual = _required_vector(lifted.dual.value, "canonical dual")
    primal_residual = data.A @ primal + slack - data.b
    dual_residual = data.A.T @ dual + data.c

    recovered = model.canonicalize().recover_numeric(primal)
    recovery = 0.0
    lower_by_id = {variable.id: variable for variable in model.lower_problem.variables()}
    for variable_id, expected in recovered.items():
        actual = lower_by_id[variable_id].value
        if actual is None:
            recovery = float("inf")
            break
        recovery = max(recovery, _norm(np.asarray(actual) - expected))
    upper = _constraint_violation(lifted.upper_constraints)
    complementarity = model.canonicalize().cone_layout.complementarity(slack, dual)
    return Residuals(
        primal_equality=_norm(primal_residual),
        dual_equality=_norm(dual_residual),
        recovery=recovery,
        upper_constraints=upper,
        primal_cone=model.canonicalize().cone_layout.primal_distance(slack),
        dual_cone=model.canonicalize().cone_layout.dual_distance(dual),
        complementarity=complementarity,
        gap_violation=max(complementarity - float(epsilon), 0.0),
    )


def _initialize_lower(
    model: BilevelProblem,
    conic_solver: str,
    options: Mapping[str, Any],
    verbose: bool,
) -> None:
    for parameter, variable in model.parameter_map.items():
        try:
            parameter.value = variable.value
        except ValueError as error:
            raise InitializationError(
                f"Upper start for {variable.name()!r} violates the declared domain "
                f"of mapped parameter {parameter.name()!r}."
            ) from error
    canonical = model.canonicalize()
    data = canonical.apply_numeric()
    primal = cp.Variable(canonical.canonical_size, name="blvpy_initial_primal")
    slack = cp.Variable(canonical.constraint_size, name="blvpy_initial_slack")
    equality = data.A @ primal + slack == data.b
    lower = cp.Problem(
        cp.Minimize(data.c @ primal),
        [equality, *canonical.cone_layout.primal_constraints(slack)],
    )
    try:
        lower.solve(solver=conic_solver, warm_start=True, verbose=verbose, **options)
    except Exception as error:
        raise InitializationError(f"The fixed-data lower cone solve failed: {error}") from error
    if lower.status not in cp.settings.SOLUTION_PRESENT:
        raise InitializationError(
            f"The fixed-data lower cone problem returned status {lower.status!r}."
        )
    if primal.value is None or slack.value is None or equality.dual_value is None:
        raise InitializationError("The conic solver omitted a primal or dual certificate.")

    lifted = model.lifted_problem
    lifted.primal.save_value(np.asarray(primal.value, dtype=float))
    lifted.slack.save_value(np.asarray(slack.value, dtype=float))
    lifted.dual.save_value(np.asarray(equality.dual_value, dtype=float))
    source_values = canonical.recover_numeric(primal.value)
    for variable in model.lower_problem.variables():
        variable.project_and_assign(source_values[variable.id])


def _restore_feasibility(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    options: Mapping[str, Any],
    verbose: bool,
    tolerance: float = 1e-7,
) -> None:
    lifted = model.lifted_problem
    radius = cp.Variable(nonneg=True, name="blvpy_restoration_radius")
    radius.value = max(1.0, _constraint_violation(lifted.problem.constraints))
    constraints: list[cp.Constraint] = []
    for constraint in lifted.upper_constraints:
        constraints.extend(_relax_constraint(constraint, radius))
    for constraint in lifted.recovery_constraints:
        constraints.extend(_relax_constraint(constraint, radius))
    constraints.extend(_relax_constraint(lifted.primal_equality, radius))
    constraints.extend(_relax_constraint(lifted.dual_equality, radius))
    constraints.extend(_relaxed_cone_constraints(model, radius))
    constraints.append(lifted.slack @ lifted.dual <= epsilon + radius)
    restoration_problem = cp.Problem(cp.Minimize(radius), constraints)
    if not restoration_problem.is_dnlp():
        raise SolveError("The feasibility-restoration problem is not DNLP compliant.")
    _solve_problem(restoration_problem, solver, options, verbose)
    if restoration_problem.status not in cp.settings.SOLUTION_PRESENT:
        raise InitializationError(
            f"Feasibility restoration returned status {restoration_problem.status!r}."
        )
    restored = compute_residuals(model, epsilon)
    if not restored.is_feasible(tolerance):
        raise InitializationError(
            "Feasibility restoration terminated without a sufficiently feasible "
            f"lifted point (max violation {restored.max_violation:.3g})."
        )


def _relaxed_cone_constraints(
    model: BilevelProblem,
    radius: cp.Expression,
) -> tuple[cp.Constraint, ...]:
    lifted = model.lifted_problem
    layout = model.canonicalize().cone_layout
    slack, dual = lifted.slack, lifted.dual
    constraints: list[cp.Constraint] = []
    if layout.zero:
        constraints.extend(
            [slack[layout.zero_slice] <= radius, -slack[layout.zero_slice] <= radius]
        )
    if layout.nonnegative:
        constraints.extend(
            [slack[layout.nonnegative_slice] >= -radius, dual[layout.nonnegative_slice] >= -radius]
        )
    for block in layout.second_order_slices:
        constraints.extend(
            [
                cp.norm(slack[block.start + 1 : block.stop], 2)
                <= slack[block.start] + radius,
                cp.norm(dual[block.start + 1 : block.stop], 2)
                <= dual[block.start] + radius,
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
    raise SolveError(
        f"Cannot construct feasibility restoration for {type(constraint).__name__}."
    )


def _solve_one(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    options: Mapping[str, Any],
    verbose: bool,
) -> IterationRecord:
    lifted = model.lifted_problem
    lifted.epsilon.value = epsilon
    message: str | None = None
    try:
        _solve_problem(lifted.problem, solver, options, verbose)
        status = lifted.problem.status or "solver_error"
    except Exception as error:
        status = "solver_error"
        message = f"{type(error).__name__}: {error}"
    try:
        residuals = compute_residuals(model, epsilon)
    except Exception:
        residuals = _infinite_residuals()
    stats = lifted.problem.solver_stats
    solve_time = getattr(stats, "solve_time", None) if stats is not None else None
    num_iters = getattr(stats, "num_iters", None) if stats is not None else None
    if not isinstance(num_iters, (int, np.integer)):
        num_iters = None
    objective = lifted.problem.value
    if objective is not None and not np.isfinite(objective):
        objective = None
    return IterationRecord(
        epsilon=epsilon,
        status=status,
        objective=objective,
        residuals=residuals,
        solver_name=str(solver),
        solve_time=solve_time,
        num_iters=num_iters,
        message=message,
    )


def _solve_problem(
    problem: cp.Problem,
    solver: str,
    options: Mapping[str, Any],
    verbose: bool,
) -> None:
    problem.solve(
        solver=solver,
        nlp=True,
        warm_start=True,
        verbose=verbose,
        **dict(options),
    )


def _compile_probe(lifted: LiftedProblem, *, use_hessian: bool) -> None:
    """Build derivative oracles after all lifted variables have initial values."""

    try:
        from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
        from cvxpy.reductions.solvers.nlp_solvers.ipopt_nlpif import IPOPT
        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Oracles

        smooth, _ = Dnlp2Smooth().apply(lifted.problem)
        data, _ = IPOPT().apply(smooth)
        oracles = Oracles(
            data["_bounds"].new_problem,
            verbose=False,
            use_hessian=use_hessian,
        )
        oracles.objective(data["x0"])
        oracles.constraints(data["x0"])
        oracles.gradient(data["x0"])
        oracles.jacobian(data["x0"])
    except Exception as error:
        raise SolveError(f"DNLP derivative compilation failed: {error}") from error


def _uses_exact_hessian(options: Mapping[str, Any]) -> bool:
    return options.get("hessian_approximation", "exact") == "exact"


def _require_solver(solver: str, *, nonlinear: bool) -> None:
    name = str(solver).upper()
    if name in {str(installed).upper() for installed in cp.installed_solvers()}:
        return
    if nonlinear and name == "IPOPT":
        raise SolverUnavailableError(
            "IPOPT is not available. Install its native library, then reinstall "
            "BLVpy so its required cyipopt binding can be built."
        )
    raise SolverUnavailableError(f"Requested solver {solver!r} is not installed in CVXPY.")


def _sampling_bounds(
    variable: cp.Variable,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    sample_bounds = getattr(variable, "sample_bounds", None)
    if sample_bounds is not None:
        try:
            lower, upper = sample_bounds
        except (TypeError, ValueError) as error:
            raise InitializationError(
                f"Variable {variable.name()!r}.sample_bounds must be a (lower, upper) pair."
            ) from error
        lower_array = np.broadcast_to(np.asarray(lower, dtype=float), variable.shape).copy()
        upper_array = np.broadcast_to(np.asarray(upper, dtype=float), variable.shape).copy()
    else:
        lower, upper = variable.get_bounds()
        lower_array = np.broadcast_to(np.asarray(lower, dtype=float), variable.shape).copy()
        upper_array = np.broadcast_to(np.asarray(upper, dtype=float), variable.shape).copy()
    fully_bounded = bool(
        np.all(np.isfinite(lower_array)) and np.all(np.isfinite(upper_array))
    )
    if np.any(lower_array > upper_array):
        raise InitializationError(
            f"Variable {variable.name()!r} has inconsistent sampling bounds."
        )
    return lower_array, upper_array, fully_bounded


def _epsilon_schedule(initial: float, target: float, contraction: float):
    epsilon = initial
    yield epsilon
    while epsilon > target:
        contracted = epsilon * contraction
        if contracted <= target or np.isclose(contracted, target, rtol=1e-12, atol=0.0):
            next_epsilon = target
        else:
            next_epsilon = contracted
        if next_epsilon >= epsilon:
            break
        epsilon = next_epsilon
        yield epsilon


def _snapshot_state(problem: cp.Problem) -> dict[int, NDArray[np.float64]]:
    state: dict[int, NDArray[np.float64]] = {}
    for variable in problem.variables():
        if variable.value is None:
            raise InitializationError(
                f"NLP solve did not return a value for variable {variable.name()!r}."
            )
        state[variable.id] = np.array(variable.value, dtype=float, copy=True)
    return state


def _restore_state(problem: cp.Problem, state: Mapping[int, ArrayLike]) -> None:
    for variable in problem.variables():
        if variable.id in state:
            variable.save_value(np.array(state[variable.id], dtype=float, copy=True))


def _assign_values(values: Mapping[cp.Variable, ArrayLike]) -> None:
    for variable, value in values.items():
        variable.project_and_assign(value)


def _result(
    model: BilevelProblem,
    status: str,
    iterations: list[IterationRecord],
    starts: list[StartRecord],
    *,
    message: str | None,
    final_record: IterationRecord | None = None,
) -> BilevelResult:
    lifted = model.lifted_problem
    variable_values = {
        variable: np.asarray(variable.value, dtype=float)
        for variable in model.source_variables
        if variable.value is not None
    }
    objective = model.outer_objective.value
    if objective is not None and not np.isfinite(objective):
        objective = None
    return BilevelResult(
        status=status,
        objective=objective,
        variable_values=variable_values,
        canonical_primal=lifted.primal.value,
        slack=lifted.slack.value,
        dual=lifted.dual.value,
        iterations=tuple(iterations),
        starts=tuple(starts),
        final_iteration=final_record,
        message=message,
        certified=False,
    )


def _record_is_acceptable(record: IterationRecord, tolerance: float) -> bool:
    return record.status in _ACCEPTABLE_NLP_STATUSES and record.residuals.is_feasible(
        tolerance
    )


def _restored_record(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    tolerance: float,
) -> IterationRecord:
    """Describe the currently restored point without invoking the NLP solver."""

    residuals = compute_residuals(model, epsilon)
    status = "optimal" if residuals.is_feasible(tolerance) else "residual_check_failed"
    value = model.outer_objective.value
    objective = None if value is None or not np.isfinite(value) else float(value)
    return IterationRecord(
        epsilon=epsilon,
        status=status,
        objective=objective,
        residuals=residuals,
        solver_name=str(solver),
        message="Diagnostics recomputed from the restored accepted point.",
    )


def _constraint_violation(constraints) -> float:
    violation = 0.0
    for constraint in constraints:
        try:
            value = np.asarray(constraint.violation(), dtype=float)
        except Exception:
            return float("inf")
        violation = max(violation, _norm(value))
    return violation


def _numeric_value(value: Any, shape: tuple[int, ...]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        array = np.reshape(array, shape, order="F")
    if not np.all(np.isfinite(array)):
        raise InitializationError("Initial upper-variable values must be finite.")
    return array.copy()


def _required_vector(value: Any, name: str) -> NDArray[np.float64]:
    if value is None:
        raise InitializationError(f"The {name} has no numeric value.")
    return np.asarray(value, dtype=float).reshape(-1, order="F")


def _norm(value: ArrayLike) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        return float("inf")
    return float(np.linalg.norm(array))


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number.") from error
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return number


def _infinite_residuals() -> Residuals:
    return Residuals(
        primal_equality=float("inf"),
        dual_equality=float("inf"),
        recovery=float("inf"),
        upper_constraints=float("inf"),
        primal_cone=float("inf"),
        dual_cone=float("inf"),
        complementarity=float("inf"),
        gap_violation=float("inf"),
    )


__all__ = ["SolveSettings", "compute_residuals", "sample_upper_starts", "solve_bilevel"]
