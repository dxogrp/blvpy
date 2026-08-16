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
    from .result import BilevelResult, GapDiagnostics


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
    """An optimistic bilevel minimization problem.

    The upper expressions reuse the lower decision-variable objects from
    ``lower_problem``. BLVPY therefore optimizes over all lower-optimal
    solutions and implements optimistic bilevel semantics.

    Parameters
    ----------
    upper_objective : cvxpy.Objective
        Objective of the upper problem. Validation requires a real-valued
        scalar :class:`cvxpy.Minimize` objective.
    lower_problem : LowerProblem
        Convex lower problem and its linked upper variables.
    upper_constraints : sequence of cvxpy.Constraint, optional
        Constraints of the upper problem. Generated constraints that preserve
        the domains of linked variables are added internally.

    Attributes
    ----------
    upper_objective : cvxpy.Objective
        The upper objective supplied at construction.
    lower_problem : LowerProblem
        The lower model supplied at construction.
    upper_constraints : tuple of cvxpy.Constraint
        The supplied upper constraints in construction order.

    Raises
    ------
    TypeError
        If an argument is not the required CVXPY or BLVPY type.

    Notes
    -----
    Constructing this object does not canonicalize or solve either level.
    Call :meth:`validate` for detailed structural diagnostics or :meth:`solve`
    to validate and compute a local numerical result.
    """

    def __init__(
        self,
        upper_objective: cp.Objective,
        lower_problem: LowerProblem,
        upper_constraints: Sequence[cp.Constraint] = (),
    ) -> None:
        if not isinstance(upper_objective, cp.Objective):
            raise TypeError("upper_objective must be a CVXPY Objective.")
        if not isinstance(lower_problem, LowerProblem):
            raise TypeError("lower_problem must be a BLVPY LowerProblem.")
        try:
            constraints = tuple(upper_constraints)
        except TypeError as error:
            raise TypeError("upper_constraints must be a sequence of CVXPY constraints.") from error
        if not all(isinstance(constraint, cp.Constraint) for constraint in constraints):
            raise TypeError("Every upper constraint must be a CVXPY Constraint.")

        self.upper_objective = upper_objective
        self.lower_problem = lower_problem
        self._cvxpy_lower_problem = lower_problem._cvxpy_problem
        self._parameter_links = lower_problem._parameter_links
        self.upper_constraints = constraints
        self._canonical: CanonicalLowerProblem | None = None
        self._lifted: _LiftedProblem | None = None

    @property
    def upper_variables(self) -> tuple[cp.Variable, ...]:
        """tuple of cvxpy.Variable: Upper variables in stable discovery order.

        Variables owned by the generated lower CVXPY problem are excluded,
        even when they also occur in the upper objective or constraints.
        """

        lower_ids = {variable.id for variable in self._cvxpy_lower_problem.variables()}
        candidates: list[cp.Variable] = []
        candidates.extend(self._parameter_links.values())
        candidates.extend(self.upper_objective.expr.variables())
        for constraint in self.upper_constraints:
            candidates.extend(constraint.variables())
        return _unique_variables(variable for variable in candidates if variable.id not in lower_ids)

    @property
    def source_variables(self) -> tuple[cp.Variable, ...]:
        """tuple of cvxpy.Variable: All user-created variables in stable order.

        Upper variables precede lower decision variables, and repeated CVXPY
        objects appear only once.
        """

        return _unique_variables((*self.upper_variables, *self._cvxpy_lower_problem.variables()))

    @property
    def _lifted_problem(self) -> _LiftedProblem:
        """Return the assembled reusable lifted problem after validation."""

        if self._lifted is None:
            self.validate()
        assert self._lifted is not None
        return self._lifted

    def is_dbp(self) -> bool:
        """Return whether the model passes BLVPY's structural validation.

        Returns
        -------
        bool
            ``True`` when :meth:`validate` succeeds, otherwise ``False``.

        Notes
        -----
        This convenience check suppresses the validation exception. Use
        :meth:`validate` when the failure reason is needed. It does not test
        numerical feasibility, boundedness, or solver availability.
        """

        try:
            self.validate()
        except Exception:
            return False
        return True

    def validate(self) -> None:
        """Validate and assemble the supported single-level reformulation.

        Validation checks the upper model, lower DCP and DPP compliance, the
        audited exact-canonicalization policy, the zero/nonnegative/SOC cone
        restriction, and DNLP compatibility of the lifted formulation.

        Returns
        -------
        None

        Raises
        ------
        ValidationError
            If the upper or lower model violates a structural requirement.
        UnsupportedModelError
            If the model uses an unsupported variable type, atom, or DNLP
            expression.
        UnsupportedConeError
            If lower canonicalization produces a cone outside SOCP mode.
        ApproximateCanonicalizationError
            If an accepted-looking source expression would be canonicalized
            only approximately.
        CanonicalizationError
            If CVXPY does not expose the expected exact conic program.

        Notes
        -----
        The canonical and lifted representations produced by successful
        validation are cached. Source-level structural checks are repeated on
        later calls. Validation does not prove feasibility, boundedness,
        constraint qualifications, or solver convergence.
        """

        self._validate_upper()
        _validate_lower(self._cvxpy_lower_problem, self._parameter_links)
        canonical = self.canonicalize()
        if self._lifted is None:
            self._lifted = self._assemble_lifted(canonical)

    def canonicalize(self) -> CanonicalLowerProblem:
        """Canonicalize the lower problem into BLVPY's affine SOCP form.

        Returns
        -------
        CanonicalLowerProblem
            Cached metadata for ``min c.T @ u + d`` subject to
            ``A @ u + s == b`` and ``s`` in the recorded product cone.

        Raises
        ------
        ValidationError
            If the lower problem is not a supported DCP/DPP minimization.
        UnsupportedConeError
            If canonicalization contains PSD, exponential, or power cones.
        CanonicalizationError
            If the fixed Clarabel-compatible reduction cannot be extracted.

        Notes
        -----
        This method is intended for numerical inspection. Fixed ordinary
        CVXPY parameters are frozen at their current values the first time
        canonicalization occurs.
        """

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
        """Solve locally by epsilon-gap continuation.

        Parameters
        ----------
        epsilon_initial : float, default=1e-1
            Positive relaxation used for the first nonlinear solve.
        epsilon_target : float, default=1e-6
            Positive final relaxation. It cannot exceed ``epsilon_initial``.
        contraction : float, default=0.1
            Factor in ``(0, 1)`` used to decrease epsilon after an accepted
            continuation point.
        best_of : int or None, default=None
            ``None`` performs one deterministic continuation. A positive
            integer performs that many independently initialized complete
            continuations and selects the acceptable target-epsilon result
            with the smallest upper objective.
        feasibility_tolerance : float, default=1e-7
            Nonnegative threshold applied to independently computed lifted
            feasibility and relaxed-gap residuals.
        seed : int, numpy.random.Generator, or None, default=None
            Random generator specification for explicit ``best_of`` runs.
            It has no effect on deterministic initialization.
        solver : str, default=cvxpy.IPOPT
            CVXPY DNLP backend used for restoration and continuation. IPOPT is
            BLVPY's installed and tested default; other CVXPY DNLP backends
            must be installed independently.
        conic_solver : str, default=cvxpy.CLARABEL
            CVXPY conic backend used for upper-point projection and fixed-upper
            lower initialization.
        solver_options : mapping or None, default=None
            Backend-specific options forwarded to each DNLP solve. The mapping
            is copied and is not modified by BLVPY.
        conic_solver_options : mapping or None, default=None
            Backend-specific options forwarded to conic solves. The mapping is
            copied and is not modified by BLVPY.
        restoration : bool, default=True
            Whether to attempt a DNLP feasibility-restoration solve when the
            initialized lifted point exceeds ``feasibility_tolerance``.
        max_retries : int, default=8
            Maximum number of intermediate-epsilon insertions following failed
            continuation attempts within each run.
        verbose : bool, default=True
            Whether to write BLVPY's concise progress transcript to standard
            error.
        solver_verbose : bool, default=False
            Whether to request CVXPY and native backend output. Backend silence
            is best effort.

        Returns
        -------
        BilevelResult
            Immutable snapshots of the selected local point, its residuals,
            and every attempted run and continuation step. If no run reaches
            the target after successful initialization, a best partial result
            with status ``"continuation_failed"`` is returned.

        Raises
        ------
        ValueError
            If a numerical setting has an invalid type or range.
        ValidationError
            If the model does not satisfy BLVPY's structural requirements.
        InitializationError
            If no run produces an acceptable initial-epsilon point.
        SolverUnavailableError
            If a requested solver is unavailable or cannot load.

        Notes
        -----
        Deterministic initialization preserves existing upper-variable values;
        otherwise it uses native-bound interior points or zero. Explicit
        ``best_of`` uses ``variable.sample_bounds`` first, then an existing
        value, then finite native bounds. Every viable run has an independent
        continuation and retry budget. The returned point is local and is not
        a global bilevel certificate.
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

    def gap_diagnostics(
        self,
        result: BilevelResult,
        *,
        solver: str = cp.CLARABEL,
        solver_options: Mapping[str, Any] | None = None,
        solver_verbose: bool = False,
    ) -> GapDiagnostics:
        """Compute canonical and source-level gap diagnostics for a result.

        Parameters
        ----------
        result : BilevelResult
            Successful or ``"continuation_failed"`` result produced by this
            problem. Complete source and canonical snapshots are required.
        solver : str, default=cvxpy.CLARABEL
            CVXPY conic backend for one fresh fixed-upper lower solve.
        solver_options : mapping or None, default=None
            Backend-specific options copied and forwarded unchanged to CVXPY.
        solver_verbose : bool, default=False
            Whether to request CVXPY and native conic-solver output.

        Returns
        -------
        GapDiagnostics
            Canonical inexact-gap terms and the signed difference between the
            returned lower objective and the reference optimum.

        Raises
        ------
        TypeError
            If ``result`` is not a :class:`blvpy.BilevelResult`.
        ValueError
            If the result is incompatible, incomplete, nonfinite, or has a
            status unsuitable for diagnosis.
        SolverUnavailableError
            If the requested conic solver is unavailable or cannot load.
        SolveError
            If the fixed-upper lower reference solve fails or returns no
            usable solution.

        Notes
        -----
        The diagnostic solve uses fixed-parameter values captured at initial
        canonicalization. All affected CVXPY variable and parameter values are
        restored before this method returns or raises. Diagnostics quantify a
        returned point; they do not certify global bilevel optimality.
        """

        from .diagnostics import _gap_diagnostics

        return _gap_diagnostics(
            self,
            result,
            solver=solver,
            solver_options=solver_options,
            solver_verbose=solver_verbose,
        )

    def _validate_upper(self) -> None:
        if not isinstance(self.upper_objective, cp.Minimize):
            raise UnsupportedModelError("The upper problem must be a minimization problem.")
        if not self.upper_objective.expr.is_scalar() or not self.upper_objective.expr.is_real():
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

        for parameter in _all_parameters(self.upper_objective, self.upper_constraints):
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
        upper_constraints = (*self.upper_constraints, *domain_constraints)
        constraints = (
            *upper_constraints,
            *recovery_constraints,
            primal_equality,
            dual_equality,
            *cone_constraints,
            gap_constraint,
        )
        problem = cp.Problem(self.upper_objective, constraints)
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
