"""Immutable solve records and numerical diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SUCCESS_STATUSES = {"optimal", "optimal_inaccurate", "success", "succeeded"}


@dataclass(frozen=True, slots=True)
class Residuals:
    """Independent residual summary for one lifted bilevel iterate.

    Parameters
    ----------
    primal_equality : float
        Euclidean norm of ``A @ u + s - b``.
    dual_equality : float
        Euclidean norm of ``A.T @ lambda + c``.
    recovery : float
        Largest Euclidean mismatch between a source lower variable and its
        value recovered from the canonical primal vector.
    upper_constraints : float
        Largest CVXPY violation norm among the upper and generated
        linked-variable domain constraints.
    primal_cone : float
        Euclidean distance from ``s`` to the primal product cone.
    dual_cone : float
        Euclidean distance from ``lambda`` to the dual product cone.
    complementarity : float
        Raw canonical pairing ``s.T @ lambda``. It may be slightly negative at
        a numerically infeasible point.
    gap_violation : float
        Violation ``max(complementarity - epsilon, 0)`` of the relaxed gap
        constraint.

    Raises
    ------
    ValueError
        If a field is not real-valued, or if a residual other than
        ``complementarity`` is negative or NaN.

    Notes
    -----
    All fields except ``complementarity`` are nonnegative. Infinite residuals
    are retained to represent missing or nonfinite numerical solver output.
    """

    primal_equality: float
    dual_equality: float
    recovery: float
    upper_constraints: float
    primal_cone: float
    dual_cone: float
    complementarity: float
    gap_violation: float

    def __post_init__(self) -> None:
        for name in (
            "primal_equality",
            "dual_equality",
            "recovery",
            "upper_constraints",
            "primal_cone",
            "dual_cone",
            "gap_violation",
        ):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "complementarity",
            _real_float(self.complementarity, "complementarity"),
        )

    @property
    def max_feasibility(self) -> float:
        """float: Largest lifted-feasibility residual, excluding the gap constraint."""

        return max(
            self.primal_equality,
            self.dual_equality,
            self.recovery,
            self.upper_constraints,
            self.primal_cone,
            self.dual_cone,
        )

    @property
    def max_violation(self) -> float:
        """float: Largest feasibility or relaxed-gap violation."""

        return max(self.max_feasibility, self.gap_violation)

    def is_feasible(self, tolerance: float, *, gap_tolerance: float | None = None) -> bool:
        """Test the lifted feasibility and relaxed-gap residuals.

        Parameters
        ----------
        tolerance : float
            Finite nonnegative bound for ``max_feasibility``.
        gap_tolerance : float or None, optional
            Finite nonnegative bound for ``gap_violation``. ``None`` uses
            ``tolerance``.

        Returns
        -------
        bool
            Whether both residual bounds are satisfied.

        Raises
        ------
        ValueError
            If either tolerance is negative, nonfinite, or not real-valued.
        """

        tolerance = _finite_nonnegative_float(tolerance, "tolerance")
        if gap_tolerance is None:
            gap_tolerance = tolerance
        else:
            gap_tolerance = _finite_nonnegative_float(gap_tolerance, "gap_tolerance")
        return self.max_feasibility <= tolerance and self.gap_violation <= gap_tolerance

    def as_dict(self) -> dict[str, float]:
        """Return all primitive residual fields as a new dictionary.

        Returns
        -------
        dict[str, float]
            Field names mapped to their stored numerical values. Derived
            properties such as ``max_violation`` are not included.
        """

        return {
            "primal_equality": self.primal_equality,
            "dual_equality": self.dual_equality,
            "recovery": self.recovery,
            "upper_constraints": self.upper_constraints,
            "primal_cone": self.primal_cone,
            "dual_cone": self.dual_cone,
            "complementarity": self.complementarity,
            "gap_violation": self.gap_violation,
        }


@dataclass(frozen=True, slots=True)
class GapDiagnostics:
    """Terms in the inexact canonical primal-dual gap identity.

    Parameters
    ----------
    primal_objective : float
        Canonical linear objective ``c.T @ u``, without the common offset.
    dual_objective : float
        Canonical dual objective ``-b.T @ lambda``, without the common offset.
    complementarity : float
        Canonical cone pairing ``s.T @ lambda``.
    dual_residual_term : float
        Correction ``u.T @ r_d``, where ``r_d = A.T @ lambda + c``.
    primal_residual_term : float
        Correction ``lambda.T @ r_p``, where ``r_p = A @ u + s - b``.
    source_gap : float or None, optional
        Signed returned lower objective minus the optimum of a fresh
        fixed-upper reference solve.
        :meth:`blvpy.BilevelProblem.gap_diagnostics` populates this field.

    Raises
    ------
    ValueError
        If any supplied diagnostic term is not real-valued.

    Notes
    -----
    The identity is

    ``primal_objective - dual_objective = complementarity +``
    ``dual_residual_term - primal_residual_term``.

    Small nonzero identity errors and slightly negative source gaps can arise
    from floating-point solver tolerances.
    """

    primal_objective: float
    dual_objective: float
    complementarity: float
    dual_residual_term: float
    primal_residual_term: float
    source_gap: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "primal_objective",
            "dual_objective",
            "complementarity",
            "dual_residual_term",
            "primal_residual_term",
        ):
            object.__setattr__(self, name, _real_float(getattr(self, name), name))
        if self.source_gap is not None:
            object.__setattr__(self, "source_gap", _real_float(self.source_gap, "source_gap"))

    @property
    def normalized_gap(self) -> float:
        """float: Canonical primal objective minus canonical dual objective."""

        return self.primal_objective - self.dual_objective

    @property
    def inexact_identity_rhs(self) -> float:
        """float: Complementarity plus the two signed residual corrections."""

        return self.complementarity + self.dual_residual_term - self.primal_residual_term

    @property
    def identity_error(self) -> float:
        """float: Left-hand side minus right-hand side of the inexact identity."""

        return self.normalized_gap - self.inexact_identity_rhs


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Numerical record for one epsilon-continuation attempt.

    Parameters
    ----------
    epsilon : float
        Finite nonnegative relaxation requested for this attempt.
    status : str
        Nonempty CVXPY solver status or BLVPY diagnostic status.
    objective : float or None
        Upper objective reported for the attempt, or ``None`` when unavailable.
    residuals : Residuals
        Independently computed lifted residuals.
    solver_name : str or None, optional
        Name of the selected nonlinear backend.
    solve_time : float or None, optional
        Finite nonnegative solver-reported time in seconds, when available.
    num_iters : int or None, optional
        Nonnegative solver-reported iteration count, when available.
    message : str or None, optional
        Additional failure or diagnostic detail.

    Raises
    ------
    ValueError
        If epsilon, status, objective, residuals, solver time, or iteration
        count has an invalid value or type.

    Notes
    -----
    A solver success status does not by itself make an attempt acceptable;
    BLVPY also applies its independent residual checks.
    """

    epsilon: float
    status: str
    objective: float | None
    residuals: Residuals
    solver_name: str | None = None
    solve_time: float | None = None
    num_iters: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _finite_nonnegative_float(self.epsilon, "epsilon"))
        _status(self.status)
        if self.objective is not None:
            object.__setattr__(self, "objective", _real_float(self.objective, "objective"))
        if not isinstance(self.residuals, Residuals):
            raise ValueError("residuals must be a Residuals instance.")
        if self.solve_time is not None:
            object.__setattr__(
                self,
                "solve_time",
                _finite_nonnegative_float(self.solve_time, "solve_time"),
            )
        if self.num_iters is not None:
            if isinstance(self.num_iters, (bool, np.bool_)) or not isinstance(self.num_iters, (int, np.integer)):
                raise ValueError("num_iters must be a nonnegative integer or None.")
            if self.num_iters < 0:
                raise ValueError("num_iters must be a nonnegative integer or None.")
            object.__setattr__(self, "num_iters", int(self.num_iters))


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Outcome and complete history of one independently initialized run.

    Parameters
    ----------
    index : int
        Zero-based nonnegative run index.
    initial_values : mapping
        Upper-variable initialization recorded for the run. These are the
        post-projection values when projection succeeds and the original
        candidate if initialization fails earlier. Keys are normally CVXPY
        variables; values are copied into read-only NumPy arrays.
    status : str
        Nonempty terminal solver or BLVPY status.
    objective : float or None, optional
        Terminal upper objective, or ``None`` when unavailable.
    iterations : tuple of IterationRecord, optional
        All epsilon-continuation attempts in execution order, including
        rejected and inserted-epsilon attempts.
    final_iteration : IterationRecord or None, optional
        Record representing the state returned for this run. If omitted while
        ``iterations`` is nonempty, the last attempted record is used.
    message : str or None, optional
        Terminal failure or diagnostic detail.

    Raises
    ------
    ValueError
        If the index, status, objective, initial-value mapping, iteration
        records, or final iteration is invalid.

    Notes
    -----
    Each explicit ``best_of`` candidate receives its own continuation and
    retry budget. A run can retain a useful partial point even when it does not
    reach the requested target epsilon.
    """

    index: int
    initial_values: Mapping[Any, ArrayLike]
    status: str
    objective: float | None = None
    iterations: tuple[IterationRecord, ...] = ()
    final_iteration: IterationRecord | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, (bool, np.bool_)) or not isinstance(self.index, (int, np.integer)):
            raise ValueError("index must be a nonnegative integer.")
        if self.index < 0:
            raise ValueError("index must be a nonnegative integer.")
        object.__setattr__(self, "index", int(self.index))
        _status(self.status)
        if self.objective is not None:
            object.__setattr__(self, "objective", _real_float(self.objective, "objective"))
        if not isinstance(self.initial_values, Mapping):
            raise ValueError("initial_values must be a mapping.")
        initial_values = {
            key: _snapshot(value, f"initial_values[{key!r}]") for key, value in self.initial_values.items()
        }
        object.__setattr__(self, "initial_values", MappingProxyType(initial_values))
        try:
            iterations = tuple(self.iterations)
        except TypeError as error:
            raise ValueError("iterations must be iterable.") from error
        if not all(isinstance(record, IterationRecord) for record in iterations):
            raise ValueError("Every iteration must be an IterationRecord.")
        object.__setattr__(self, "iterations", iterations)
        final_iteration = self.final_iteration
        if final_iteration is not None and not isinstance(final_iteration, IterationRecord):
            raise ValueError("final_iteration must be an IterationRecord or None.")
        if final_iteration is None and iterations:
            final_iteration = iterations[-1]
        object.__setattr__(self, "final_iteration", final_iteration)

    @property
    def epsilon_history(self) -> tuple[float, ...]:
        """tuple of float: Successful tolerances in decreasing accepted order."""

        return _accepted_epsilon_history(self.iterations)

    @property
    def attempted_epsilon_history(self) -> tuple[float, ...]:
        """tuple of float: All attempted tolerances, including failures and retries."""

        return tuple(record.epsilon for record in self.iterations)

    @property
    def solver_statuses(self) -> tuple[str, ...]:
        """tuple of str: Status of every attempt in continuation order."""

        return tuple(record.status for record in self.iterations)

    @property
    def residuals(self) -> Residuals | None:
        """Residuals or None: Residuals at this run's returned point."""

        return self.final_iteration.residuals if self.final_iteration is not None else None

    @property
    def complementarity(self) -> float | None:
        """float or None: Canonical complementarity at the returned point."""

        residuals = self.residuals
        return None if residuals is None else residuals.complementarity

    @property
    def final_epsilon(self) -> float | None:
        """float or None: Continuation tolerance at the returned point."""

        return self.final_iteration.epsilon if self.final_iteration is not None else None

    @property
    def succeeded(self) -> bool:
        """bool: Whether the terminal status is one of BLVPY's success statuses."""

        return self.status.lower() in _SUCCESS_STATUSES


@dataclass(frozen=True, slots=True)
class BilevelResult:
    """Immutable result returned by :meth:`blvpy.BilevelProblem.solve`.

    Parameters
    ----------
    status : str
        Nonempty terminal status. Use :attr:`succeeded` instead of depending
        on backend-specific success spellings.
    objective : float or None, optional
        Upper objective at the returned point, or ``None`` when unavailable.
    variable_values : mapping, optional
        Snapshots keyed by the original upper and lower CVXPY variable objects.
    canonical_primal : array-like or None, optional
        Returned canonical lower primal vector ``u``.
    slack : array-like or None, optional
        Returned canonical primal cone vector ``s``.
    dual : array-like or None, optional
        Returned canonical dual cone vector ``lambda``.
    iterations : tuple of IterationRecord, optional
        Continuation attempts belonging to the selected run.
    runs : tuple of RunRecord, optional
        Every deterministic or explicit ``best_of`` run in index order.
    selected_run_index : int or None, optional
        Zero-based index of the run whose state is exposed by the top-level
        fields.
    final_iteration : IterationRecord or None, optional
        Record representing the returned selected-run state. If omitted while
        ``iterations`` is nonempty, the last attempted record is used.
    message : str or None, optional
        Terminal failure or diagnostic detail.

    Raises
    ------
    ValueError
        If a scalar field, numerical snapshot, iteration or run collection,
        selected-run index, or final iteration is invalid or inconsistent.

    Notes
    -----
    Numeric values are copied into read-only NumPy arrays, and mappings are
    read-only views. A ``"continuation_failed"`` result exposes the best
    available partial run but is not successful.
    """

    status: str
    objective: float | None = None
    variable_values: Mapping[Any, ArrayLike] = field(default_factory=dict)
    canonical_primal: ArrayLike | None = None
    slack: ArrayLike | None = None
    dual: ArrayLike | None = None
    iterations: tuple[IterationRecord, ...] = ()
    runs: tuple[RunRecord, ...] = ()
    selected_run_index: int | None = None
    final_iteration: IterationRecord | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        _status(self.status)
        if self.objective is not None:
            object.__setattr__(self, "objective", _real_float(self.objective, "objective"))
        if not isinstance(self.variable_values, Mapping):
            raise ValueError("variable_values must be a mapping.")
        values = {key: _snapshot(value, f"variable_values[{key!r}]") for key, value in self.variable_values.items()}
        object.__setattr__(self, "variable_values", MappingProxyType(values))
        for name in ("canonical_primal", "slack", "dual"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _snapshot(value, name))

        try:
            iterations = tuple(self.iterations)
            runs = tuple(self.runs)
        except TypeError as error:
            raise ValueError("iterations and runs must be iterable.") from error
        if not all(isinstance(record, IterationRecord) for record in iterations):
            raise ValueError("Every iteration must be an IterationRecord.")
        if not all(isinstance(record, RunRecord) for record in runs):
            raise ValueError("Every run must be a RunRecord.")
        run_indices = tuple(record.index for record in runs)
        if len(set(run_indices)) != len(run_indices):
            raise ValueError("Run indices must be unique.")
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "runs", runs)
        selected_run_index = self.selected_run_index
        if selected_run_index is not None:
            if isinstance(selected_run_index, (bool, np.bool_)) or not isinstance(
                selected_run_index, (int, np.integer)
            ):
                raise ValueError("selected_run_index must be a nonnegative integer or None.")
            selected_run_index = int(selected_run_index)
            if selected_run_index < 0:
                raise ValueError("selected_run_index must be a nonnegative integer or None.")
            if selected_run_index not in run_indices:
                raise ValueError("selected_run_index must identify one of the recorded runs.")
        object.__setattr__(self, "selected_run_index", selected_run_index)
        final_iteration = self.final_iteration
        if final_iteration is not None and not isinstance(final_iteration, IterationRecord):
            raise ValueError("final_iteration must be an IterationRecord or None.")
        if final_iteration is None and iterations:
            final_iteration = iterations[-1]
        object.__setattr__(self, "final_iteration", final_iteration)

    @property
    def epsilon_history(self) -> tuple[float, ...]:
        """tuple of float: Selected-run tolerances accepted in decreasing order."""

        return _accepted_epsilon_history(self.iterations)

    @property
    def attempted_epsilon_history(self) -> tuple[float, ...]:
        """tuple of float: All selected-run attempts, including failures and retries."""

        return tuple(record.epsilon for record in self.iterations)

    @property
    def solver_statuses(self) -> tuple[str, ...]:
        """tuple of str: Status of every selected-run continuation attempt."""

        return tuple(record.status for record in self.iterations)

    @property
    def residuals(self) -> Residuals | None:
        """Residuals or None: Independent residuals at the returned point."""

        return self.final_iteration.residuals if self.final_iteration is not None else None

    @property
    def complementarity(self) -> float | None:
        """float or None: Canonical complementarity at the returned point."""

        residuals = self.residuals
        return None if residuals is None else residuals.complementarity

    @property
    def final_epsilon(self) -> float | None:
        """float or None: Epsilon associated with the returned point."""

        return self.final_iteration.epsilon if self.final_iteration is not None else None

    @property
    def succeeded(self) -> bool:
        """bool: Whether the top-level status is a BLVPY success status."""

        return self.status.lower() in _SUCCESS_STATUSES

    @property
    def selected_run(self) -> RunRecord | None:
        """RunRecord or None: Run exposed by the top-level result fields."""

        if self.selected_run_index is None:
            return None
        return next(record for record in self.runs if record.index == self.selected_run_index)

    @property
    def all_objectives(self) -> tuple[float | None, ...]:
        """tuple: Terminal objective of every run in recorded order."""

        return tuple(record.objective for record in self.runs)


def _accepted_epsilon_history(iterations: tuple[IterationRecord, ...]) -> tuple[float, ...]:
    accepted: list[float] = []
    for record in iterations:
        if record.status.lower() not in _SUCCESS_STATUSES:
            continue
        if not accepted or record.epsilon < accepted[-1]:
            accepted.append(record.epsilon)
    return tuple(accepted)


def _status(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("status must be a nonempty string.")


def _real_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real number.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real number.") from error
    return result


def _nonnegative_float(value: object, name: str) -> float:
    result = _real_float(value, name)
    if np.isnan(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative and not NaN.")
    return result


def _finite_nonnegative_float(value: object, name: str) -> float:
    result = _nonnegative_float(value, name)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _snapshot(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    try:
        snapshot = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    snapshot.setflags(write=False)
    return snapshot
