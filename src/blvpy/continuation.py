"""Initialization, restoration, and epsilon-gap continuation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import sqrt
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cones.power import _power_3d_dual_scale as _power_3d_dual_scale
from ._continuation.residuals import (
    _constraint_violation,
    _finite_constraint_violation,
    _infinite_residuals,
)
from ._continuation.residuals import (
    _norm as _norm,
)
from ._continuation.residuals import (
    _required_vector as _required_vector,
)
from ._continuation.residuals import compute_residuals as _compute_residuals
from ._continuation.restoration import _relax_constraint, _relaxed_cone_constraints
from ._continuation.sampling import _generate_upper_initializations
from ._continuation.sampling import (
    _project_variable_value as _project_variable_value,
)
from ._continuation.sampling import (
    _validated_sample_bounds as _validated_sample_bounds,
)
from ._continuation.sampling import (
    _variable_bounds as _variable_bounds,
)
from ._continuation.state import (
    _assign_values,
    _numeric_value,
    _restore_state,
    _snapshot_state,
    _sync_linked_parameters,
)
from .backends import solve_conic, solve_dnlp
from .errors import InitializationError, SolveError, SolverUnavailableError
from .fixed_lower import FixedLowerSolveError, solve_fixed_lower
from .progress import ProgressReporter
from .result import BilevelResult, IterationRecord, Residuals, RunRecord

if TYPE_CHECKING:
    from .problem import BilevelProblem, _LiftedProblem

_ACCEPTABLE_NLP_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(frozen=True, slots=True)
class _RunOutcome:
    record: RunRecord
    state: Mapping[int, NDArray[np.float64]] | None
    accepted_initial: bool
    reached_target: bool


@dataclass(frozen=True, slots=True)
class _SolveSettings:
    """Numerical settings for deterministic or best-of gap continuation."""

    epsilon_initial: float = 1e-1
    epsilon_target: float = 1e-6
    contraction: float = 0.1
    best_of: int | None = None
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


def solve_bilevel(model: BilevelProblem, settings: _SolveSettings) -> BilevelResult:
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
    successful_runs = sum(record.succeeded for record in result.runs)
    attempted_records = tuple(record for run in result.runs for record in run.iterations)
    accepted_attempts = sum(_record_is_acceptable(record, tolerance) for record in attempted_records)
    reporter.summary(
        result=result,
        successful_runs=successful_runs,
        requested_runs=1 if settings.best_of is None else int(settings.best_of),
        accepted_attempts=accepted_attempts,
        attempted_solves=len(attempted_records),
        elapsed=perf_counter() - started_at,
    )
    return result


def _solve_bilevel(
    model: BilevelProblem,
    settings: _SolveSettings,
    reporter: ProgressReporter,
    solver_verbose: bool,
) -> BilevelResult:
    """Execute deterministic or best-of complete continuation runs."""

    if settings.solver is None:
        raise ValueError("solver must name a CVXPY DNLP backend; None is not supported.")

    epsilon_initial = _finite_nonnegative(settings.epsilon_initial, "epsilon_initial")
    epsilon_target = _finite_nonnegative(settings.epsilon_target, "epsilon_target")
    if epsilon_initial <= 0 or epsilon_target <= 0:
        raise ValueError("epsilon_initial and epsilon_target must be positive.")
    if epsilon_target > epsilon_initial:
        raise ValueError("epsilon_target cannot exceed epsilon_initial.")

    contraction = _finite_nonnegative(settings.contraction, "contraction")
    if not 0 < contraction < 1:
        raise ValueError("contraction must lie strictly between zero and one.")

    best_of = settings.best_of
    if best_of is not None:
        if isinstance(best_of, (bool, np.bool_)) or not isinstance(best_of, (int, np.integer)) or best_of < 1:
            raise ValueError("best_of must be a positive integer or None.")
        best_of = int(best_of)

    max_retries = settings.max_retries
    if isinstance(max_retries, (bool, np.bool_)) or not isinstance(max_retries, (int, np.integer)) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer.")

    feasibility_tolerance = _finite_nonnegative(
        settings.feasibility_tolerance,
        "feasibility_tolerance",
    )
    solver_options = dict(settings.solver_options or {})
    conic_solver_options = dict(settings.conic_solver_options or {})
    requested_runs = 1 if best_of is None else best_of

    model.validate()
    lifted = model._lifted_problem
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
        exp=layout.exponential,
        power_3d=layout.power_3d,
        lower_solver=str(settings.conic_solver),
        nonlinear_solver=str(settings.solver),
        best_of=best_of,
        requested_runs=requested_runs,
        epsilon_initial=epsilon_initial,
        epsilon_target=epsilon_target,
        contraction=contraction,
    )

    rng = settings.seed if isinstance(settings.seed, np.random.Generator) else np.random.default_rng(settings.seed)
    variables_without_values = tuple(variable for variable in model.upper_variables if variable.value is None)
    try:
        initializations = _generate_upper_initializations(model, best_of, rng)
    except InitializationError as error:
        if best_of is None:
            raise _automatic_initialization_error(
                model,
                variables_without_values,
                str(error),
            ) from error
        raise
    reporter.initialization()

    outcomes: list[_RunOutcome] = []
    for index, raw_initialization in enumerate(initializations):
        recorded_initialization = {
            variable: _numeric_value(value, variable.shape) for variable, value in raw_initialization.items()
        }
        reporter.run_begin(
            run_index=index,
            total_runs=requested_runs,
            randomized=best_of is not None,
        )
        try:
            _assign_values(raw_initialization)
            before_projection = _constraint_violation(lifted.upper_constraints)
            if before_projection > 0.0:
                reporter.projection(
                    run_index=index,
                    total_runs=requested_runs,
                    before_violation=before_projection,
                )
            initialization = _project_upper_start(
                model,
                raw_initialization,
                settings.conic_solver,
                conic_solver_options,
                solver_verbose,
            )
            recorded_initialization = {
                variable: _numeric_value(value, variable.shape) for variable, value in initialization.items()
            }
            outcome = _initialize_run(
                model,
                index,
                requested_runs,
                initialization,
                epsilon_initial=epsilon_initial,
                feasibility_tolerance=feasibility_tolerance,
                solver=settings.solver,
                conic_solver=settings.conic_solver,
                solver_options=solver_options,
                conic_solver_options=conic_solver_options,
                restoration=settings.restoration,
                solver_verbose=solver_verbose,
                reporter=reporter,
            )
        except SolverUnavailableError:
            raise
        except Exception as error:
            record = RunRecord(
                index=index,
                initial_values=recorded_initialization,
                status="initialization_failed",
                message=f"{type(error).__name__}: {error}",
            )
            outcome = _RunOutcome(record, None, False, False)
            reporter.run(
                run_index=index,
                total_runs=requested_runs,
                record=record,
            )
        outcomes.append(outcome)

    initialized = [outcome for outcome in outcomes if outcome.accepted_initial]
    if not initialized:
        details = "; ".join(
            f"run {outcome.record.index + 1}: {outcome.record.message or outcome.record.status}" for outcome in outcomes
        )
        if best_of is None:
            raise _automatic_initialization_error(
                model,
                variables_without_values,
                details,
            )
        error = InitializationError("All best-of runs failed at the initial epsilon.")
        if details:
            error.add_note(f"Run details: {details}")
        raise error

    reporter.continuation()
    for position, outcome in enumerate(outcomes):
        if not outcome.accepted_initial:
            continue
        try:
            completed = _solve_run(
                model,
                outcome,
                total_runs=requested_runs,
                epsilon_initial=epsilon_initial,
                epsilon_target=epsilon_target,
                contraction=contraction,
                feasibility_tolerance=feasibility_tolerance,
                solver=settings.solver,
                solver_options=solver_options,
                solver_verbose=solver_verbose,
                max_retries=int(max_retries),
                reporter=reporter,
            )
        except SolverUnavailableError:
            raise
        except Exception as error:
            failed = RunRecord(
                index=outcome.record.index,
                initial_values=outcome.record.initial_values,
                status="continuation_failed",
                objective=outcome.record.objective,
                iterations=outcome.record.iterations,
                final_iteration=outcome.record.final_iteration,
                message=f"{type(error).__name__}: {error}",
            )
            completed = _RunOutcome(failed, outcome.state, True, False)
        outcomes[position] = completed
        reporter.run(
            run_index=completed.record.index,
            total_runs=requested_runs,
            record=completed.record,
        )

    successful = [
        outcome
        for outcome in outcomes
        if outcome.reached_target
        and outcome.record.succeeded
        and outcome.record.objective is not None
        and np.isfinite(outcome.record.objective)
    ]
    if successful:
        selected = min(
            successful,
            key=lambda outcome: (
                _upper_objective_merit(model, float(outcome.record.objective)),
                outcome.record.index,
            ),
        )
        result_status = selected.record.status
        result_message = selected.record.message
    else:
        partial = [outcome for outcome in outcomes if outcome.accepted_initial and outcome.state is not None]
        selected = min(partial, key=lambda outcome: _partial_run_key(model, outcome))
        result_status = "continuation_failed"
        result_message = "No run reached the requested target epsilon with acceptable residuals."

    assert selected.state is not None
    _restore_state(lifted.problem, selected.state)
    _sync_linked_parameters(model)
    final = selected.record.final_iteration
    if final is not None:
        lifted.epsilon.value = final.epsilon
    reporter.selected_run(
        run_index=selected.record.index,
        total_runs=requested_runs,
        objective=selected.record.objective,
    )
    return _result(
        model,
        result_status,
        list(selected.record.iterations),
        [outcome.record for outcome in outcomes],
        selected_run_index=selected.record.index,
        objective=selected.record.objective,
        message=result_message,
        final_record=final,
        feasibility_tolerance=feasibility_tolerance,
    )


def _initialize_run(
    model: BilevelProblem,
    index: int,
    total_runs: int,
    initial_values: Mapping[cp.Variable, ArrayLike],
    *,
    epsilon_initial: float,
    feasibility_tolerance: float,
    solver: str,
    conic_solver: str,
    solver_options: Mapping[str, Any],
    conic_solver_options: Mapping[str, Any],
    restoration: bool,
    solver_verbose: bool,
    reporter: ProgressReporter,
) -> _RunOutcome:
    """Initialize and solve one run at the initial continuation tolerance."""

    normalized_initial_values = {
        variable: _numeric_value(value, variable.shape) for variable, value in initial_values.items()
    }
    _assign_values(normalized_initial_values)
    _initialize_lower(
        model,
        conic_solver,
        conic_solver_options,
        solver_verbose,
    )
    lifted = model._lifted_problem
    lifted.epsilon.value = epsilon_initial
    initial_residuals = compute_residuals(model, epsilon_initial)
    if restoration and not initial_residuals.is_feasible(feasibility_tolerance):
        reporter.restoration(
            run_index=index,
            total_runs=total_runs,
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

    _compile_probe(lifted)
    record = _checked_record(
        _solve_one(
            model,
            epsilon_initial,
            solver,
            solver_options,
            solver_verbose,
        ),
        feasibility_tolerance,
    )
    accepted = _record_is_acceptable(record, feasibility_tolerance)
    reporter.attempt(
        run_index=index,
        total_runs=total_runs,
        attempt_index=0,
        kind="initial",
        epsilon=epsilon_initial,
        record=record,
        accepted=accepted,
    )
    run = RunRecord(
        index=index,
        initial_values=normalized_initial_values,
        status=record.status,
        objective=record.objective,
        iterations=(record,),
        final_iteration=record,
        message=record.message,
    )
    if not accepted:
        reporter.run(run_index=index, total_runs=total_runs, record=run)
        return _RunOutcome(run, None, False, False)
    return _RunOutcome(
        run,
        _snapshot_state(lifted.problem),
        True,
        False,
    )


def _solve_run(
    model: BilevelProblem,
    initial: _RunOutcome,
    *,
    total_runs: int,
    epsilon_initial: float,
    epsilon_target: float,
    contraction: float,
    feasibility_tolerance: float,
    solver: str,
    solver_options: Mapping[str, Any],
    solver_verbose: bool,
    max_retries: int,
    reporter: ProgressReporter,
) -> _RunOutcome:
    """Complete one independent epsilon continuation from its saved state."""

    if not initial.accepted_initial or initial.state is None:
        return initial

    lifted = model._lifted_problem
    _restore_state(lifted.problem, initial.state)
    _sync_linked_parameters(model)
    iterations = list(initial.record.iterations)
    last_successful_epsilon = epsilon_initial
    last_successful_state = dict(initial.state)
    targets: list[tuple[float, str]] = [
        (epsilon, "scheduled")
        for epsilon in _epsilon_schedule(
            epsilon_initial,
            epsilon_target,
            contraction,
        )
    ][1:]
    target_index = 0
    retries = 0
    attempt_index = 1

    while target_index < len(targets):
        epsilon, kind = targets[target_index]
        _restore_state(lifted.problem, last_successful_state)
        _sync_linked_parameters(model)
        record = _checked_record(
            _solve_one(
                model,
                epsilon,
                solver,
                solver_options,
                solver_verbose,
            ),
            feasibility_tolerance,
        )
        iterations.append(record)
        accepted = _record_is_acceptable(record, feasibility_tolerance)
        reporter.attempt(
            run_index=initial.record.index,
            total_runs=total_runs,
            attempt_index=attempt_index,
            kind=kind,
            epsilon=epsilon,
            record=record,
            accepted=accepted,
        )
        attempt_index += 1

        if accepted:
            last_successful_epsilon = epsilon
            last_successful_state = _snapshot_state(lifted.problem)
            target_index += 1
            if kind == "scheduled":
                retries = 0
            continue

        if retries >= max_retries:
            reporter.retry_exhausted(
                run_index=initial.record.index,
                total_runs=total_runs,
                last_successful_epsilon=last_successful_epsilon,
                max_retries=max_retries,
            )
            return _partial_run_outcome(
                model,
                initial.record,
                iterations,
                last_successful_state,
                last_successful_epsilon,
                solver,
                feasibility_tolerance,
                f"Could not reduce epsilon below {last_successful_epsilon:.6g}.",
            )

        intermediate = sqrt(last_successful_epsilon * epsilon)
        if not epsilon < intermediate < last_successful_epsilon:
            return _partial_run_outcome(
                model,
                initial.record,
                iterations,
                last_successful_state,
                last_successful_epsilon,
                solver,
                feasibility_tolerance,
                "Continuation could not insert a distinct intermediate epsilon.",
            )
        targets.insert(target_index, (intermediate, "inserted"))
        retries += 1
        reporter.inserting_epsilon(
            run_index=initial.record.index,
            total_runs=total_runs,
            epsilon=intermediate,
            target=epsilon,
        )

    _restore_state(lifted.problem, last_successful_state)
    _sync_linked_parameters(model)
    lifted.epsilon.value = last_successful_epsilon
    try:
        final = _restored_record(
            model,
            last_successful_epsilon,
            solver,
            feasibility_tolerance,
        )
    except Exception as error:
        final = _diagnostic_failure_record(
            model,
            last_successful_epsilon,
            solver,
            error,
        )

    status = final.status
    message = final.message
    if not final.residuals.is_feasible(
        feasibility_tolerance,
        gap_tolerance=feasibility_tolerance,
    ):
        status = "residual_check_failed"
        message = message or (
            "The NLP solver returned a point, but one or more independently "
            "computed residuals exceed the requested feasibility tolerance."
        )
        final = replace(final, status=status, message=message)

    objective = final.objective
    if status in _ACCEPTABLE_NLP_STATUSES and (objective is None or not np.isfinite(objective)):
        status = "objective_unavailable"
        message = "The target-epsilon point has no finite upper objective."
        final = replace(final, status=status, message=message)
    reached_target = (
        np.isclose(
            last_successful_epsilon,
            epsilon_target,
            rtol=1e-12,
            atol=0.0,
        )
        and status in _ACCEPTABLE_NLP_STATUSES
        and objective is not None
        and np.isfinite(objective)
    )
    run = RunRecord(
        index=initial.record.index,
        initial_values=initial.record.initial_values,
        status=status,
        objective=objective,
        iterations=tuple(iterations),
        final_iteration=final,
        message=message,
    )
    return _RunOutcome(
        run,
        dict(last_successful_state),
        True,
        reached_target,
    )


def _partial_run_outcome(
    model: BilevelProblem,
    initial_record: RunRecord,
    iterations: list[IterationRecord],
    state: Mapping[int, ArrayLike],
    epsilon: float,
    solver: str,
    tolerance: float,
    message: str,
) -> _RunOutcome:
    lifted = model._lifted_problem
    _restore_state(lifted.problem, state)
    _sync_linked_parameters(model)
    lifted.epsilon.value = epsilon
    try:
        final = _restored_record(model, epsilon, solver, tolerance)
    except Exception as error:
        final = _diagnostic_failure_record(model, epsilon, solver, error)
    run = RunRecord(
        index=initial_record.index,
        initial_values=initial_record.initial_values,
        status="continuation_failed",
        objective=final.objective,
        iterations=tuple(iterations),
        final_iteration=final,
        message=message,
    )
    return _RunOutcome(run, dict(state), True, False)


def _partial_run_key(
    model: BilevelProblem,
    outcome: _RunOutcome,
) -> tuple[float, float, int]:
    epsilon = outcome.record.final_epsilon
    objective = outcome.record.objective
    return (
        float("inf") if epsilon is None else float(epsilon),
        (
            float("inf")
            if objective is None or not np.isfinite(objective)
            else _upper_objective_merit(model, float(objective))
        ),
        outcome.record.index,
    )


def _upper_objective_merit(model: BilevelProblem, objective: float) -> float:
    """Return a minimization-oriented key for an original-sense objective."""

    return -objective if isinstance(model.upper_objective, cp.Maximize) else objective


def _checked_record(
    record: IterationRecord,
    tolerance: float,
) -> IterationRecord:
    if record.status not in _ACCEPTABLE_NLP_STATUSES:
        return record
    if record.residuals.is_feasible(tolerance):
        return record
    return replace(
        record,
        status="residual_check_failed",
        message=(f"Independent residual check failed: max violation {record.residuals.max_violation:.3g}."),
    )


def _project_upper_start(
    model: BilevelProblem,
    sample: Mapping[cp.Variable, ArrayLike],
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> dict[cp.Variable, NDArray[np.float64]]:
    """Best-effort projection of the deterministic start onto upper constraints."""

    projected = {variable: _numeric_value(value, variable.shape) for variable, value in sample.items()}
    constraints = model._lifted_problem.upper_constraints
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

    return _compute_residuals(model, epsilon)


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
    try:
        solution = solve_fixed_lower(model, conic_solver, options, solver_verbose)
    except (cp.SolverError, FixedLowerSolveError) as error:
        raise InitializationError(f"The fixed-data lower cone solve failed: {error}") from error

    lifted = model._lifted_problem
    lifted.primal.save_value(np.asarray(solution.primal, dtype=float))
    lifted.slack.save_value(np.asarray(solution.slack, dtype=float))
    lifted.dual.save_value(np.asarray(solution.dual, dtype=float))
    for variable in model._cvxpy_lower_problem.variables():
        variable.project_and_assign(solution.source_values[variable.id])


def _restore_feasibility(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
    tolerance: float = 1e-7,
) -> None:
    lifted = model._lifted_problem
    radius = cp.Variable(nonneg=True, name="blvpy_restoration_radius")
    initial_violation = _constraint_violation(lifted.problem.constraints)
    if not np.isfinite(initial_violation):
        initial_violation = _finite_constraint_violation(lifted.problem.constraints)
    radius.value = max(1.0, initial_violation)
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


def _solve_one(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    options: Mapping[str, Any],
    solver_verbose: bool,
) -> IterationRecord:
    lifted = model._lifted_problem
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


def _compile_probe(lifted: _LiftedProblem) -> None:
    """Compile and evaluate solver-neutral first-order DNLP oracles."""

    try:
        from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
        from cvxpy.reductions.flip_objective import FlipObjective
        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Bounds, Oracles

        problem = lifted.problem
        if isinstance(problem.objective, cp.Maximize):
            problem, _ = FlipObjective().apply(problem)
        smooth, _ = Dnlp2Smooth().apply(problem)
        bounds = Bounds(smooth)
        oracles = Oracles(
            bounds.new_problem,
            verbose=False,
            use_hessian=False,
        )
        oracles.objective(bounds.x0)
        oracles.constraints(bounds.x0)
        oracles.gradient(bounds.x0)
        oracles.jacobian(bounds.x0)
    except Exception as error:
        raise SolveError(f"DNLP derivative compilation failed: {error}") from error


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


def _automatic_initialization_error(
    model: BilevelProblem,
    variables_without_values: tuple[cp.Variable, ...],
    details: str,
) -> InitializationError:
    variables = variables_without_values or model.upper_variables
    names = ", ".join(variable.name() for variable in variables)
    if names:
        message = f"Automatic initialization failed. Please initialize variables: {names}."
    else:
        message = "Automatic initialization failed."
    error = InitializationError(message)
    if details:
        error.add_note(f"Initialization details: {details}")
    return error


def _result(
    model: BilevelProblem,
    status: str,
    iterations: list[IterationRecord],
    runs: list[RunRecord],
    *,
    selected_run_index: int,
    objective: float | None,
    message: str | None,
    final_record: IterationRecord | None = None,
    feasibility_tolerance: float = 1e-7,
) -> BilevelResult:
    lifted = model._lifted_problem
    variable_values = {
        variable: np.asarray(variable.value, dtype=float)
        for variable in model.source_variables
        if variable.value is not None
    }
    return BilevelResult(
        status=status,
        objective=objective,
        variable_values=variable_values,
        canonical_primal=lifted.primal.value,
        slack=lifted.slack.value,
        dual=lifted.dual.value,
        iterations=tuple(iterations),
        runs=tuple(runs),
        selected_run_index=selected_run_index,
        final_iteration=final_record,
        message=message,
        _feasibility_tolerance=feasibility_tolerance,
        _problem_token=model._result_token,
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
    value = model.upper_objective.value
    objective = None if value is None or not np.isfinite(value) else float(value)
    return IterationRecord(
        epsilon=epsilon,
        status=status,
        objective=objective,
        residuals=residuals,
        solver_name=str(solver),
        message=None,
    )


def _diagnostic_failure_record(
    model: BilevelProblem,
    epsilon: float,
    solver: str,
    error: Exception,
) -> IterationRecord:
    """Fail closed when the restored point cannot be diagnosed independently."""

    try:
        value = model.upper_objective.value
        objective = None if value is None or not np.isfinite(value) else float(value)
    except Exception:
        objective = None
    return IterationRecord(
        epsilon=epsilon,
        status="residual_check_failed",
        objective=objective,
        residuals=_infinite_residuals(),
        solver_name=str(solver),
        message=f"Could not recompute restored-point diagnostics: {type(error).__name__}: {error}",
    )


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


__all__ = ["compute_residuals"]
