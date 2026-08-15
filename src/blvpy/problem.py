"""Public bilevel modeling interface.

The lower problem is canonicalized once. Its source variables stay in the
upper model and are linked to the canonical primal variable through CVXPY's
affine inverse map, giving optimistic bilevel semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np

from .canonicalization import (
    CanonicalExpressions,
    CanonicalLowerProblem,
    _canonicalize_lower,
    _validate_lower,
)
from .errors import UnsupportedModelError, ValidationError
from .lower_problem import LowerProblem

if TYPE_CHECKING:
    from .result import BilevelResult


@dataclass(frozen=True, slots=True)
class _LiftedProblem:
    """Internal reusable epsilon-parameterized DNLP reformulation."""

    problem: cp.Problem
    epsilon: cp.Parameter
    primal: cp.Variable
    slack: cp.Variable
    dual: cp.Variable
    canonical_expressions: CanonicalExpressions
    recovery_expressions: Mapping[int, cp.Expression]
    upper_constraints: tuple[cp.Constraint, ...]
    recovery_constraints: tuple[cp.Constraint, ...]
    primal_equality: cp.Constraint
    dual_equality: cp.Constraint
    cone_constraints: tuple[cp.Constraint, ...]
    gap_constraint: cp.Constraint

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recovery_expressions",
            MappingProxyType(dict(self.recovery_expressions)),
        )


class BilevelProblem:
    """An optimistic bilevel problem with a DPP convex lower problem."""

    def __init__(
        self,
        outer_objective: cp.Objective,
        lower_problem: LowerProblem,
        outer_constraints: Sequence[cp.Constraint] = (),
    ) -> None:
        if not isinstance(outer_objective, cp.Objective):
            raise TypeError("outer_objective must be a CVXPY Objective.")
        if not isinstance(lower_problem, LowerProblem):
            raise TypeError("lower_problem must be a BLVPY LowerProblem.")
        try:
            constraints = tuple(outer_constraints)
        except TypeError as error:
            raise TypeError("outer_constraints must be a sequence of CVXPY constraints.") from error
        if not all(isinstance(constraint, cp.Constraint) for constraint in constraints):
            raise TypeError("Every outer constraint must be a CVXPY Constraint.")

        self.outer_objective = outer_objective
        self.lower_problem = lower_problem
        self._cvxpy_lower_problem = lower_problem._cvxpy_problem
        self._parameter_links = lower_problem._parameter_links
        self.outer_constraints = constraints
        self._canonical: CanonicalLowerProblem | None = None
        self._lifted: _LiftedProblem | None = None

    @property
    def upper_variables(self) -> tuple[cp.Variable, ...]:
        """Upper variables, excluding source variables owned by the lower problem."""

        lower_ids = {variable.id for variable in self._cvxpy_lower_problem.variables()}
        candidates: list[cp.Variable] = []
        candidates.extend(self._parameter_links.values())
        candidates.extend(self.outer_objective.expr.variables())
        for constraint in self.outer_constraints:
            candidates.extend(constraint.variables())
        return _unique_variables(variable for variable in candidates if variable.id not in lower_ids)

    @property
    def source_variables(self) -> tuple[cp.Variable, ...]:
        """All user-created upper and lower variables in stable order."""

        return _unique_variables((*self.upper_variables, *self._cvxpy_lower_problem.variables()))

    @property
    def _lifted_problem(self) -> _LiftedProblem:
        """Return the assembled reusable lifted problem after validation."""

        if self._lifted is None:
            self.validate()
        assert self._lifted is not None
        return self._lifted

    def is_dbp(self) -> bool:
        """Return whether the problem passes structural and DNLP validation."""

        try:
            self.validate()
        except Exception:
            return False
        return True

    def validate(self) -> None:
        """Validate the lower canonicalization and complete lifted DNLP model."""

        self._validate_outer()
        _validate_lower(self._cvxpy_lower_problem, self._parameter_links)
        canonical = self.canonicalize()
        if self._lifted is None:
            self._lifted = self._assemble_lifted(canonical)

    def canonicalize(self) -> CanonicalLowerProblem:
        """Return immutable metadata for the fixed Clarabel SOCP reduction."""

        if self._canonical is None:
            self._canonical = _canonicalize_lower(
                self._cvxpy_lower_problem,
                self._parameter_links,
            )
        return self._canonical

    def solve(
        self,
        *,
        epsilon_initial: float = 1e-1,
        epsilon_target: float = 1e-6,
        contraction: float = 0.1,
        best_of: int | None = None,
        feasibility_tolerance: float = 1e-7,
        seed: int | np.random.Generator | None = None,
        solver: str = cp.IPOPT,
        conic_solver: str = cp.CLARABEL,
        solver_options: Mapping[str, Any] | None = None,
        conic_solver_options: Mapping[str, Any] | None = None,
        restoration: bool = True,
        max_retries: int = 8,
        verbose: bool = True,
        solver_verbose: bool = False,
    ) -> BilevelResult:
        """Solve locally with deterministic or best-of epsilon continuation.

        IPOPT is the default nonlinear backend. Solver options are passed
        through CVXPY; returned feasibility and gap diagnostics are computed
        independently from the solver status. ``verbose`` controls BLVPY's
        progress transcript, while ``solver_verbose`` controls CVXPY and
        native solver output. Explicit ``best_of=N`` runs ``N`` independently
        initialized complete continuations and returns the best acceptable
        target-epsilon result.
        """

        from .continuation import _SolveSettings, solve_bilevel

        settings = _SolveSettings(
            epsilon_initial=epsilon_initial,
            epsilon_target=epsilon_target,
            contraction=contraction,
            best_of=best_of,
            feasibility_tolerance=feasibility_tolerance,
            seed=seed,
            solver=solver,
            conic_solver=conic_solver,
            solver_options=solver_options,
            conic_solver_options=conic_solver_options,
            restoration=restoration,
            max_retries=max_retries,
            verbose=verbose,
            solver_verbose=solver_verbose,
        )
        return solve_bilevel(self, settings)

    def _validate_outer(self) -> None:
        if not isinstance(self.outer_objective, cp.Minimize):
            raise UnsupportedModelError("The upper problem must be a minimization problem.")
        if not self.outer_objective.expr.is_scalar() or not self.outer_objective.expr.is_real():
            raise ValidationError("The upper objective must be a real-valued scalar expression.")

        lower_ids = {variable.id for variable in self._cvxpy_lower_problem.variables()}
        lower_parameter_ids = {parameter.id for parameter in self._cvxpy_lower_problem.parameters()}
        for parameter, variable in self._parameter_links.items():
            if not isinstance(parameter, cp.Parameter):
                raise ValidationError("Every generated lower parameter must be a CVXPY Parameter.")
            if parameter.id not in lower_parameter_ids:
                raise ValidationError(f"Generated parameter {parameter.name()!r} does not occur in the lower problem.")
            if not isinstance(variable, cp.Variable):
                raise ValidationError(f"Linked value for {parameter.name()!r} must be a CVXPY Variable.")
            if variable.id in lower_ids:
                raise ValidationError(f"Linked variable {variable.name()!r} is also a lower-level variable.")
            if parameter.shape != variable.shape:
                raise ValidationError(
                    f"Generated parameter {parameter.name()!r} has shape {parameter.shape}, "
                    f"but variable {variable.name()!r} has shape {variable.shape}."
                )
            if not variable.is_real():
                raise UnsupportedModelError(f"Linked upper variable {variable.name()!r} must be real-valued.")

        for parameter in _all_parameters(self.outer_objective, self.outer_constraints):
            if parameter.id not in lower_parameter_ids and parameter.value is None:
                raise ValidationError(f"Unmapped upper parameter {parameter.name()!r} must have a value.")
        for variable in self.source_variables:
            attributes = variable.attributes
            if attributes.get("boolean") or attributes.get("integer"):
                raise UnsupportedModelError(
                    f"Variable {variable.name()!r} is mixed-integer; continuous variables are required."
                )
            if not variable.is_real():
                raise UnsupportedModelError(f"Variable {variable.name()!r} is complex; real variables are required.")

    def _assemble_lifted(self, canonical: CanonicalLowerProblem) -> _LiftedProblem:
        parameter_expressions = {parameter.id: variable for parameter, variable in self._parameter_links.items()}
        data = canonical.build_data_expressions(parameter_expressions)
        primal = cp.Variable(canonical.canonical_size, name="blvpy_primal")
        slack = cp.Variable(canonical.constraint_size, name="blvpy_slack")
        dual = cp.Variable(canonical.constraint_size, name="blvpy_dual")
        epsilon = cp.Parameter(nonneg=True, value=1e-1, name="blvpy_epsilon")

        recovery = canonical.recovery_expressions(primal)
        lower_by_id = {variable.id: variable for variable in self._cvxpy_lower_problem.variables()}
        recovery_constraints = tuple(
            lower_by_id[variable_id] == expression for variable_id, expression in recovery.items()
        )
        primal_equality = data.A @ primal + slack == data.b
        dual_equality = data.A.T @ dual + data.c == 0
        cone_constraints = (
            *canonical.cone_layout.primal_constraints(slack),
            *canonical.cone_layout.dual_constraints(dual),
        )
        gap_constraint = slack @ dual <= epsilon
        domain_constraints = _linked_parameter_domain_constraints(self._parameter_links)
        upper_constraints = (*self.outer_constraints, *domain_constraints)
        constraints = (
            *upper_constraints,
            *recovery_constraints,
            primal_equality,
            dual_equality,
            *cone_constraints,
            gap_constraint,
        )
        problem = cp.Problem(self.outer_objective, constraints)
        if not problem.is_dnlp():
            atoms = sorted({atom.__name__ for atom in problem.atoms()})
            detail = ", ".join(atoms) if atoms else "unknown expression"
            raise UnsupportedModelError(
                f"The assembled optimistic reformulation is not DNLP compliant; encountered atoms: {detail}."
            )

        try:
            from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth

            Dnlp2Smooth().apply(problem)
        except Exception as error:
            raise UnsupportedModelError(f"DNLP could not compile the lifted problem: {error}") from error

        return _LiftedProblem(
            problem=problem,
            epsilon=epsilon,
            primal=primal,
            slack=slack,
            dual=dual,
            canonical_expressions=data,
            recovery_expressions=recovery,
            upper_constraints=tuple(upper_constraints),
            recovery_constraints=recovery_constraints,
            primal_equality=primal_equality,
            dual_equality=dual_equality,
            cone_constraints=tuple(cone_constraints),
            gap_constraint=gap_constraint,
        )


def _unique_variables(variables) -> tuple[cp.Variable, ...]:
    unique: dict[int, cp.Variable] = {}
    for variable in variables:
        unique.setdefault(variable.id, variable)
    return tuple(unique.values())


def _all_parameters(
    objective: cp.Objective,
    constraints: Sequence[cp.Constraint],
) -> tuple[cp.Parameter, ...]:
    parameters: dict[int, cp.Parameter] = {parameter.id: parameter for parameter in objective.expr.parameters()}
    for constraint in constraints:
        parameters.update({parameter.id: parameter for parameter in constraint.parameters()})
    return tuple(parameters.values())


def _linked_parameter_domain_constraints(
    linked_parameters: Mapping[cp.Parameter, cp.Variable],
) -> tuple[cp.Constraint, ...]:
    """Carry parameter-domain assumptions into the upper-variable model."""

    constraints: list[cp.Constraint] = []
    for parameter, variable in linked_parameters.items():
        attributes = parameter.attributes
        if attributes.get("complex") or attributes.get("imag"):
            raise UnsupportedModelError(f"Generated parameter {parameter.name()!r} must be real-valued.")
        if attributes.get("boolean") or attributes.get("integer"):
            raise UnsupportedModelError(f"Generated parameter {parameter.name()!r} has a discrete domain.")
        if attributes.get("PSD") or attributes.get("NSD") or attributes.get("hermitian"):
            raise UnsupportedModelError(
                f"Generated parameter {parameter.name()!r} requires a matrix cone outside SOCP mode."
            )
        if attributes.get("nonneg") or attributes.get("pos"):
            constraints.append(variable >= 0)
        if attributes.get("nonpos") or attributes.get("neg"):
            constraints.append(variable <= 0)
        if attributes.get("symmetric"):
            constraints.append(variable == variable.T)
        if attributes.get("diag"):
            constraints.append(variable == cp.diag(cp.diag(variable)))

        lower, upper = parameter.get_bounds()
        lower = np.broadcast_to(np.asarray(lower, dtype=float), parameter.shape or ())
        upper = np.broadcast_to(np.asarray(upper, dtype=float), parameter.shape or ())
        vector = cp.vec(variable, order="F") if variable.ndim else cp.reshape(variable, (1,), order="F")
        lower_vector = np.asarray(lower).reshape(-1, order="F")
        upper_vector = np.asarray(upper).reshape(-1, order="F")
        lower_indices = np.flatnonzero(np.isfinite(lower_vector))
        upper_indices = np.flatnonzero(np.isfinite(upper_vector))
        if lower_indices.size and not (attributes.get("nonneg") or attributes.get("pos")):
            constraints.append(vector[lower_indices] >= lower_vector[lower_indices])
        if upper_indices.size and not (attributes.get("nonpos") or attributes.get("neg")):
            constraints.append(vector[upper_indices] <= upper_vector[upper_indices])
    return tuple(constraints)


__all__ = ["BilevelProblem"]
