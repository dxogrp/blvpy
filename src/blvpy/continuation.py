"""Initialization, restoration, and epsilon-gap continuation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import solve_conic, solve_dnlp
from .errors import InitializationError, SolveError, SolverUnavailableError
from .progress import ProgressReporter
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
    starts: int = 1
    feasibility_tolerance: float = 1e-7
    seed: int | np.random.Generator | None = None
    solver: str = cp.IPOPT
    conic_solver: str = cp.CLARABEL
    solver_options: Mapping[str, Any] | None = None
    conic_solver_options: Mapping[str, Any] | None = None
    restoration: bool = True
    max_retries: int = 8
    verbose: bool = True
    solver_verbose: bool = False


def solve_bilevel(model: BilevelProblem, settings: SolveSettings) -> BilevelResult:
    """Internal implementation of :meth:`BilevelProblem.solve`."""

    progress_verbose = _boolean(settings.verbose, "verbose")
    reporter = ProgressReporter(enabled=progress_verbose)
    started_at = perf_counter()
    try:
        solver_verbose = _boolean(settings.solver_verbose, "solver_verbose")
        result = _solve_bilevel(model, settings, reporter, solver_verbose)
    except Exception as error:
        reporter.failure(error, elapsed=perf_counter() - started_at)
        raise

    tolerance = float(settings.feasibility_tolerance)
    successful_starts = sum(
        record.status in _ACCEPTABLE_NLP_STATUSES
        and record.residuals is not None
        and record.residuals.is_feasible(tolerance)
        for record in result.starts
    )
    continuation_records = result.iterations[1:]
    accepted_attempts = sum(_record_is_acceptable(record, tolerance) for record in continuation_records)
    reporter.summary(
        result=result,
        successful_starts=successful_starts,
        requested_starts=int(settings.starts),
        accepted_attempts=accepted_attempts,
        attempted_solves=len(continuation_records),
        elapsed=perf_counter() - started_at,
    )
    return result


def _solve_bilevel(
    model: BilevelProblem,
    settings: SolveSettings,
    reporter: ProgressReporter,
    solver_verbose: bool,
) -> BilevelResult:
    """Execute one solve while emitting semantic lifecycle progress."""

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
    feasibility_tolerance = _finite_nonnegative(feasibility_tolerance, "feasibility_tolerance")
    solver_options = dict(solver_options or {})
    conic_solver_options = dict(conic_solver_options or {})

    model.validate()
    lifted = model.lifted_problem
    canonical = model.canonicalize()
    layout = canonical.cone_layout
    reporter.problem(
        upper_dimension=sum(variable.size for variable in model.upper_variables),
        lower_dimension=sum(variable.size for variable in model._cvxpy_lower_problem.variables()),
        canonical_variables=canonical.canonical_size,
        canonical_constraints=canonical.constraint_size,
        zero=layout.zero,
        nonnegative=layout.nonnegative,
        soc=layout.second_order,
        lower_solver=str(conic_solver),
        nonlinear_solver=str(solver),
        requested_starts=int(starts),
        epsilon_initial=epsilon_initial,
        epsilon_target=epsilon_target,
        contraction=contraction,
    )
    reporter.initialization()
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    variables_without_values = tuple(variable for variable in model.upper_variables if variable.value is None)
    try:
        samples = list(sample_upper_starts(model, int(starts), rng))
        if samples:
            _assign_values(samples[0])
            before_projection = _constraint_violation(lifted.upper_constraints)
            if before_projection > 0.0:
                reporter.projection(
                    start_index=0,
                    total_starts=len(samples),
                    before_violation=before_projection,
                )
            samples[0] = _project_upper_start(
                model,
                samples[0],
                conic_solver,
                conic_solver_options,
                solver_verbose,
            )
            samples = list(_deduplicate_starts(samples))
    except InitializationError as error:
        raise _automatic_initialization_error(model, variables_without_values, str(error)) from error

    reporter.starts(
        requested_starts=int(starts),
        deduplicated_starts=len(samples),
    )

    start_records: list[StartRecord] = []
    candidates: list[_Candidate] = []
    for index, sample in enumerate(samples):
        try:
            _assign_values(sample)
            _initialize_lower(model, conic_solver, conic_solver_options, solver_verbose)
            lifted.epsilon.value = epsilon_initial
            initial_residuals = compute_residuals(model, epsilon_initial)
            if restoration and not initial_residuals.is_feasible(feasibility_tolerance):
                reporter.restoration(
                    start_index=index,
                    total_starts=len(samples),
                    before_violation=initial_residuals.max_violation,
                )
                _restore_feasibility(
                    model,
                    epsilon_initial,
                    solver,
                    solver_options,
                    solver_verbose,
                    feasibility_tolerance,
                )
            _compile_probe(lifted, use_hessian=_uses_exact_hessian(solver_options))
            record = _solve_one(
                model,
                epsilon_initial,
                solver,
                solver_options,
                solver_verbose,
            )
            if not _record_is_acceptable(record, feasibility_tolerance):
                message = record.message
                if record.status in _ACCEPTABLE_NLP_STATUSES:
                    message = f"Independent residual check failed: max violation {record.residuals.max_violation:.3g}."
                start_record = StartRecord(
                    index,
                    "residual_check_failed" if record.status in _ACCEPTABLE_NLP_STATUSES else record.status,
                    record.objective,
                    record.residuals,
                    message,
                )
                start_records.append(start_record)
                reporter.start(
                    start_index=index,
                    total_starts=len(samples),
                    record=start_record,
                    accepted=False,
                    solve_time=record.solve_time,
                    num_iters=record.num_iters,
                )
                continue
            state = _snapshot_state(lifted.problem)
            objective = float(record.objective) if record.objective is not None else float("inf")
            candidate = _Candidate(index, objective, state, record.residuals)
            candidates.append(candidate)
            start_record = StartRecord(index, record.status, record.objective, record.residuals)
            start_records.append(start_record)
            reporter.start(
                start_index=index,
                total_starts=len(samples),
                record=start_record,
                accepted=True,
                solve_time=record.solve_time,
                num_iters=record.num_iters,
            )
        except SolverUnavailableError as error:
            reporter.start(
                start_index=index,
                total_starts=len(samples),
                record=StartRecord(index, "failed", message=f"{type(error).__name__}: {error}"),
                accepted=False,
            )
            raise
        except Exception as error:
            start_record = StartRecord(index, "failed", message=f"{type(error).__name__}: {error}")
            start_records.append(start_record)
            reporter.start(
                start_index=index,
                total_starts=len(samples),
                record=start_record,
                accepted=False,
            )

    if not candidates:
        details = "; ".join(f"start {record.index}: {record.message or record.status}" for record in start_records)
        raise _automatic_initialization_error(model, variables_without_values, details)

    candidates.sort(key=lambda candidate: candidate.objective)
    best = candidates[0]
    _restore_state(lifted.problem, best.state)
    reporter.selected_start(
        start_index=best.index,
        total_starts=len(samples),
        objective=best.objective,
    )
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
    attempt_index = 0
    reporter.continuation()
    while target_index < len(targets):
        epsilon = targets[target_index]
        _restore_state(lifted.problem, last_successful_state)
        record = _solve_one(model, epsilon, solver, solver_options, solver_verbose)
        iterations.append(record)
        accepted = _record_is_acceptable(record, feasibility_tolerance)
        reporter.attempt(
            attempt_index=attempt_index,
            kind="scheduled" if epsilon in scheduled_targets else "inserted",
            epsilon=epsilon,
            record=record,
            accepted=accepted,
        )
        attempt_index += 1
        if accepted:
            last_successful_epsilon = epsilon
            last_successful_state = _snapshot_state(lifted.problem)
            target_index += 1
            if epsilon in scheduled_targets:
                retries = 0
            continue

        recovered = False
        for alternative in alternatives:
            _restore_state(lifted.problem, alternative.state)
            retry_record = _solve_one(model, epsilon, solver, solver_options, solver_verbose)
            iterations.append(retry_record)
            retry_accepted = _record_is_acceptable(retry_record, feasibility_tolerance)
            reporter.attempt(
                attempt_index=attempt_index,
                kind="alternative-start",
                epsilon=epsilon,
                record=retry_record,
                accepted=retry_accepted,
                start_index=alternative.index,
            )
            attempt_index += 1
            if retry_accepted:
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
            reporter.retry_exhausted(
                last_successful_epsilon=last_successful_epsilon,
                max_retries=int(max_retries),
            )
            _restore_state(lifted.problem, last_successful_state)
            lifted.epsilon.value = last_successful_epsilon
            return _result(
                model,
                "continuation_failed",
                iterations,
                start_records,
                message=f"Could not reduce epsilon below {last_successful_epsilon:.6g}.",
                final_record=_restored_record(model, last_successful_epsilon, solver, feasibility_tolerance),
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
                final_record=_restored_record(model, last_successful_epsilon, solver, feasibility_tolerance),
            )
        targets.insert(target_index, intermediate)
        retries += 1
        reporter.inserting_epsilon(epsilon=intermediate, target=epsilon)

    _restore_state(lifted.problem, last_successful_state)
    lifted.epsilon.value = last_successful_epsilon
    try:
        final = _restored_record(model, last_successful_epsilon, solver, feasibility_tolerance)
    except Exception:
        final = next(record for record in reversed(iterations) if _record_is_acceptable(record, feasibility_tolerance))
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
    """Generate deterministic and optional randomized upper starts.

    Existing values take precedence for the first start. Otherwise, each
    component uses the midpoint of two finite bounds, or zero clipped to a
    finite one-sided bound. Additional starts randomize only components with
    two finite bounds.
    """

    specifications: list[
        tuple[
            cp.Variable,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.bool_],
        ]
    ] = []
    for variable in model.upper_variables:
        lower, upper = _variable_bounds(variable)
        bounded = np.isfinite(lower) & np.isfinite(upper)
        if variable.value is not None:
            deterministic = _numeric_value(variable.value, variable.shape)
        else:
            deterministic = np.zeros(variable.shape, dtype=float)
            deterministic[bounded] = (lower[bounded] + upper[bounded]) / 2.0
            deterministic = np.where(
                np.isfinite(lower) & ~np.isfinite(upper),
                np.maximum(deterministic, lower),
                deterministic,
            )
            deterministic = np.where(
                ~np.isfinite(lower) & np.isfinite(upper),
                np.minimum(deterministic, upper),
                deterministic,
            )
            deterministic = _project_variable_value(variable, deterministic)
        specifications.append((variable, deterministic, lower, upper, bounded))

    deterministic_sample = {variable: deterministic.copy() for variable, deterministic, _, _, _ in specifications}
    samples: list[dict[cp.Variable, NDArray[np.float64]]] = [deterministic_sample]
    for _ in range(1, starts):
        sample: dict[cp.Variable, NDArray[np.float64]] = {}
        for variable, deterministic, lower, upper, bounded in specifications:
            value = deterministic.copy()
            if np.any(bounded):
                value[bounded] = rng.uniform(lower[bounded], upper[bounded])
                value = _project_variable_value(variable, value)
            sample[variable] = value
        samples.append(sample)
    return _deduplicate_starts(samples)


def _project_upper_start(
    model: BilevelProblem,
    sample: Mapping[cp.Variable, ArrayLike],
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> dict[cp.Variable, NDArray[np.float64]]:
    """Best-effort projection of the deterministic start onto upper constraints."""

    projected = {variable: _numeric_value(value, variable.shape) for variable, value in sample.items()}
    constraints = model.lifted_problem.upper_constraints
    if not constraints:
        return projected
    _assign_values(projected)
    if _constraint_violation(constraints) == 0.0:
        return projected

    distance = sum(
        (cp.sum_squares(variable - value) if variable.ndim else cp.square(variable - value))
        for variable, value in projected.items()
    )
    projection = cp.Problem(cp.Minimize(distance), constraints)
    if not projection.is_dcp():
        return projected
    try:
        solve_conic(projection, solver, options, solver_verbose)
    except cp.SolverError:
        _assign_values(projected)
        return projected
    if projection.status not in cp.settings.SOLUTION_PRESENT:
        _assign_values(projected)
        return projected
    try:
        return {variable: _numeric_value(variable.value, variable.shape) for variable in projected}
    except InitializationError:
        _assign_values(projected)
        return projected


def compute_residuals(model: BilevelProblem, epsilon: float | None = None) -> Residuals:
    """Independently compute all reported lifted residuals."""

    lifted = model.lifted_problem
    if epsilon is None:
        epsilon = float(lifted.epsilon.value)
    values = {parameter: variable.value for parameter, variable in model._parameter_links.items()}
    if any(value is None for value in values.values()):
        raise InitializationError("Linked upper variables do not all have numeric values.")
    data = model.canonicalize().apply_numeric(values)
    primal = _required_vector(lifted.primal.value, "canonical primal")
    slack = _required_vector(lifted.slack.value, "canonical slack")
    dual = _required_vector(lifted.dual.value, "canonical dual")
    primal_residual = data.A @ primal + slack - data.b
    dual_residual = data.A.T @ dual + data.c

    recovered = model.canonicalize().recover_numeric(primal)
    recovery = 0.0
    lower_by_id = {variable.id: variable for variable in model._cvxpy_lower_problem.variables()}
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
    solver_verbose: bool,
) -> None:
    for parameter, variable in model._parameter_links.items():
        try:
            parameter.value = variable.value
        except ValueError as error:
            raise InitializationError(
                f"Upper start for {variable.name()!r} violates the declared domain "
                f"of generated lower parameter {parameter.name()!r}."
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
        solve_conic(lower, conic_solver, options, solver_verbose)
    except cp.SolverError as error:
        raise InitializationError(f"The fixed-data lower cone solve failed: {error}") from error
    if lower.status not in cp.settings.SOLUTION_PRESENT:
        raise InitializationError(f"The fixed-data lower cone problem returned status {lower.status!r}.")
    if primal.value is None or slack.value is None or equality.dual_value is None:
        raise InitializationError("The conic solver omitted a primal or dual certificate.")

    lifted = model.lifted_problem
    lifted.primal.save_value(np.asarray(primal.value, dtype=float))
    lifted.slack.save_value(np.asarray(slack.value, dtype=float))
    lifted.dual.save_value(np.asarray(equality.dual_value, dtype=float))
    source_values = canonical.recover_numeric(primal.value)
    for variable in model._cvxpy_lower_problem.variables():
        variable.project_and_assign(source_values[variable.id])


def _restore_feasibility(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
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
    solve_dnlp(restoration_problem, solver, options, solver_verbose)
    if restoration_problem.status not in cp.settings.SOLUTION_PRESENT:
        raise InitializationError(f"Feasibility restoration returned status {restoration_problem.status!r}.")
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


def _solve_one(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> IterationRecord:
    lifted = model.lifted_problem
    lifted.epsilon.value = epsilon
    message: str | None = None
    try:
        solve_dnlp(lifted.problem, solver, options, solver_verbose)
        status = lifted.problem.status or "solver_error"
    except SolverUnavailableError:
        raise
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


def _variable_bounds(
    variable: cp.Variable,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower, upper = variable.get_bounds()
    lower_array = np.broadcast_to(np.asarray(lower, dtype=float), variable.shape).copy()
    upper_array = np.broadcast_to(np.asarray(upper, dtype=float), variable.shape).copy()
    if np.any(lower_array > upper_array):
        raise InitializationError(f"Variable {variable.name()!r} has inconsistent bounds.")
    return lower_array, upper_array


def _project_variable_value(
    variable: cp.Variable,
    value: ArrayLike,
) -> NDArray[np.float64]:
    try:
        projected = variable.project(value)
    except (TypeError, ValueError) as error:
        raise InitializationError(
            f"Could not project an automatic value for upper variable {variable.name()!r} onto its declared attributes."
        ) from error
    if hasattr(projected, "toarray"):
        projected = projected.toarray()
    return _numeric_value(projected, variable.shape)


def _deduplicate_starts(
    samples: list[Mapping[cp.Variable, ArrayLike]],
) -> tuple[dict[cp.Variable, NDArray[np.float64]], ...]:
    unique: list[dict[cp.Variable, NDArray[np.float64]]] = []
    keys: set[tuple[tuple[int, bytes], ...]] = set()
    for sample in samples:
        normalized = {variable: _numeric_value(value, variable.shape) for variable, value in sample.items()}
        key = tuple((variable.id, np.ascontiguousarray(value).tobytes()) for variable, value in normalized.items())
        if key not in keys:
            keys.add(key)
            unique.append(normalized)
    return tuple(unique)


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
            raise InitializationError(f"NLP solve did not return a value for variable {variable.name()!r}.")
        state[variable.id] = np.array(variable.value, dtype=float, copy=True)
    return state


def _restore_state(problem: cp.Problem, state: Mapping[int, ArrayLike]) -> None:
    for variable in problem.variables():
        if variable.id in state:
            variable.save_value(np.array(state[variable.id], dtype=float, copy=True))


def _assign_values(values: Mapping[cp.Variable, ArrayLike]) -> None:
    for variable, value in values.items():
        variable.project_and_assign(value)


def _automatic_initialization_error(
    model: BilevelProblem,
    variables_without_values: tuple[cp.Variable, ...],
    details: str,
) -> InitializationError:
    variables = variables_without_values or model.upper_variables
    names = ", ".join(variable.name() for variable in variables)
    error = InitializationError(f"Automatic initialization failed. Please initialize variables: {names}.")
    if details:
        error.add_note(f"Initialization details: {details}")
    return error


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
    return record.status in _ACCEPTABLE_NLP_STATUSES and record.residuals.is_feasible(tolerance)


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


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean.")
    return bool(value)


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
