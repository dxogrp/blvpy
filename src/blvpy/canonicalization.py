"""Lossless DPP-to-SOCP canonicalization for lower-level problems.

This module deliberately stops at CVXPY's pre-solver conic representation.
The returned matrices use the solver convention ``A @ u + s == b``; no
Clarabel scaling or presolve data is involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy import settings as cvxpy_settings
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.elementwise.abs import abs as abs_atom
from cvxpy.atoms.elementwise.huber import huber
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.power import PowerApprox
from cvxpy.atoms.geo_mean import GeoMeanApprox
from cvxpy.atoms.max import max as max_atom
from cvxpy.atoms.min import min as min_atom
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.pnorm import PnormApprox
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.reductions.solution import Solution
from numpy.typing import ArrayLike, NDArray

from .cones import ConeLayout
from .errors import (
    ApproximateCanonicalizationError,
    CanonicalizationError,
    ParameterMappingError,
    UnsupportedConeError,
    UnsupportedModelError,
    ValidationError,
)

ParameterTransform = Literal["identity", "symmetric", "diagonal", "sparse"]

# This is intentionally narrower than “anything CVXPY can turn into an SOCP”.
# Each nonlinear entry below has an exact epigraph/hypograph graph whose
# pointwise projection is preserved by CVXPY's Dcp2Cone reduction.  Affine
# atoms are audited as a class because their graph and recovery are identities.
_AUDITED_NONLINEAR_ATOMS = frozenset(
    {
        abs_atom,
        huber,
        max_atom,
        maximum,
        min_atom,
        minimum,
        norm1,
        norm_inf,
        PnormApprox,
        PowerApprox,
        QuadForm,
        quad_over_lin,
        sum_largest,
    }
)
_AUDITED_REDUCTION_CHAIN = (
    "Dcp2Cone",
    "CvxAttr2Constr",
    "EliminateZeroSized",
    "ConeMatrixStuffing",
    "CLARABEL",
)


@dataclass(frozen=True, slots=True)
class CanonicalData:
    """Numerical data for ``min c.T @ u + d`` with ``A @ u + s == b``."""

    A: sp.csc_array
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    d: float

    def __post_init__(self) -> None:
        matrix = sp.csc_array(self.A, dtype=float)
        b = _readonly_vector(self.b)
        c = _readonly_vector(self.c)
        if matrix.shape != (b.size, c.size):
            raise ValueError(f"Canonical A has shape {matrix.shape}; expected {(b.size, c.size)}.")
        if not np.isfinite(matrix.data).all() or not np.isfinite(b).all() or not np.isfinite(c).all():
            raise ValueError("Canonical data must be finite.")
        d = float(np.asarray(self.d).reshape(()))
        if not np.isfinite(d):
            raise ValueError("Canonical objective offset must be finite.")
        object.__setattr__(self, "A", matrix)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)


@dataclass(frozen=True, slots=True)
class AffineRecoveryMap:
    """Immutable affine recovery for all original lower variables."""

    specs: tuple[RecoverySpec, ...]

    def expressions(self, u: cp.Expression) -> dict[int, cp.Expression]:
        """Return shaped recovery expressions keyed by source variable ID."""

        return {spec.variable_id: spec.expression(u) for spec in self.specs}

    def numeric(self, u: ArrayLike) -> dict[int, NDArray[np.float64]]:
        """Return recovered numeric values keyed by source variable ID."""

        return {spec.variable_id: spec.numeric(u) for spec in self.specs}


@dataclass(frozen=True, slots=True)
class CanonicalExpressions:
    """CVXPY expressions for a canonical lower problem's affine data."""

    A: cp.Expression
    b: cp.Expression
    c: cp.Expression
    d: cp.Expression


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """How one original lower parameter enters CVXPY's parameter vector."""

    parameter_id: int
    name: str
    shape: tuple[int, ...]
    size: int
    mapped: bool
    internal_parameter_id: int
    internal_shape: tuple[int, ...]
    internal_size: int
    offset: int
    transform: ParameterTransform = "identity"
    sparse_indices: tuple[tuple[int, ...], ...] = ()

    def pack_numeric(self, value: ArrayLike) -> NDArray[np.float64]:
        """Pack an original-shaped value in CVXPY's internal Fortran order."""

        array = np.asarray(value, dtype=float)
        if array.shape != self.shape:
            raise ParameterMappingError(f"Parameter {self.name!r} expects shape {self.shape}, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ParameterMappingError(f"Parameter {self.name!r} must have a finite value.")
        if self.transform == "identity":
            packed = array.reshape(-1, order="F")
        elif self.transform == "symmetric":
            packed = array[np.triu_indices(self.shape[-1])]
        elif self.transform == "diagonal":
            packed = np.diag(array)
        else:
            packed = array[self.sparse_indices]
        return np.asarray(packed, dtype=float).reshape(-1, order="F")

    def pack_expression(self, value: cp.Expression) -> cp.Expression:
        """Symbolically pack an original-shaped affine expression."""

        expression = cp.Expression.cast_to_const(value)
        if expression.shape != self.shape:
            raise ParameterMappingError(f"Parameter {self.name!r} expects shape {self.shape}, got {expression.shape}.")
        if not expression.is_affine():
            raise ParameterMappingError(f"Expression linked to parameter {self.name!r} must be affine.")
        if self.transform == "identity":
            return cp.reshape(expression, (self.internal_size,), order="F")
        if self.transform == "symmetric":
            rows, columns = np.triu_indices(self.shape[-1])
            return cp.hstack([expression[row, column] for row, column in zip(rows, columns)])
        if self.transform == "diagonal":
            return cp.diag(expression)
        return cp.hstack([expression[index] for index in zip(*self.sparse_indices)])


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    """A fixed affine map from canonical ``u`` to one source variable."""

    variable_id: int
    name: str
    shape: tuple[int, ...]
    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != int(np.prod(self.shape, dtype=int)):
            raise ValueError("Invalid source-variable recovery matrix shape.")
        if offset.shape != (matrix.shape[0],):
            raise ValueError("Invalid source-variable recovery offset shape.")
        matrix.setflags(write=False)
        offset.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    def expression(self, u: cp.Expression) -> cp.Expression:
        """Recover a shaped source-variable expression from canonical ``u``."""

        vector = cp.reshape(cp.Expression.cast_to_const(u), (self.matrix.shape[1],), order="F")
        flat = self.matrix @ vector + self.offset
        return cp.reshape(flat, self.shape, order="F")

    def numeric(self, u: ArrayLike) -> NDArray[np.float64]:
        """Recover a shaped source-variable value from canonical ``u``."""

        vector = np.asarray(u, dtype=float).reshape(-1)
        if vector.size != self.matrix.shape[1]:
            raise ValueError(f"Canonical vector has length {vector.size}; expected {self.matrix.shape[1]}.")
        return np.asarray(self.matrix @ vector + self.offset).reshape(self.shape, order="F")


@dataclass(frozen=True, slots=True)
class _DataAffineMap:
    """Coefficients of canonical data against CVXPY's packed parameter vector."""

    A: tuple[sp.csc_array, ...]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    d: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CanonicalLowerProblem:
    """A fixed exact SOCP canonicalization of a CVXPY lower problem."""

    lower_problem: cp.Problem
    canonical_problem: cp.Problem
    parameter_map: Mapping[cp.Parameter, cp.Expression]
    param_prog: Any
    reduction_chain: tuple[Any, ...]
    inverse_data: tuple[Any, ...]
    canonical_variable_offsets: Mapping[int, int]
    cone_layout: ConeLayout
    canonical_size: int
    constraint_size: int
    parameter_specs: tuple[ParameterSpec, ...]
    recovery_specs: tuple[RecoverySpec, ...]
    _affine_map: _DataAffineMap
    fixed_parameter_values: Mapping[int, NDArray[np.float64]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_map", MappingProxyType(dict(self.parameter_map)))
        object.__setattr__(
            self,
            "canonical_variable_offsets",
            MappingProxyType(dict(self.canonical_variable_offsets)),
        )
        fixed_values: dict[int, NDArray[np.float64]] = {}
        for parameter_id, value in self.fixed_parameter_values.items():
            snapshot = np.array(value, dtype=float, copy=True)
            snapshot.setflags(write=False)
            fixed_values[int(parameter_id)] = snapshot
        object.__setattr__(self, "fixed_parameter_values", MappingProxyType(fixed_values))

    @property
    def source_variable_ids(self) -> tuple[int, ...]:
        """Original lower-variable IDs in CVXPY problem order."""

        return tuple(spec.variable_id for spec in self.recovery_specs)

    @property
    def recovery_map(self) -> AffineRecoveryMap:
        """Fixed affine recovery map for the source lower variables."""

        return AffineRecoveryMap(self.recovery_specs)

    @property
    def parameter_ids(self) -> tuple[int, ...]:
        """Mapped lower-parameter IDs in CVXPY problem order.

        Unmapped parameters are frozen into constant canonical data at
        canonicalization time and therefore do not appear here.
        """

        return tuple(spec.parameter_id for spec in self.parameter_specs)

    def apply_numeric(
        self,
        values: Mapping[cp.Parameter | int, ArrayLike] | None = None,
    ) -> CanonicalData:
        """Evaluate canonical data using overrides and current fixed values.

        ``values`` may be keyed by original Parameter objects or IDs. Missing
        mapped parameters are read from their linked expression. Unmapped
        parameters were frozen at canonicalization time.
        """

        overrides = _normalise_value_keys(values or {})
        packed = self._parameter_vector_numeric(overrides)
        return self._evaluate_affine_map(packed)

    def build_data_expressions(
        self,
        parameter_expr_by_id: Mapping[cp.Parameter | int, cp.Expression],
    ) -> CanonicalExpressions:
        """Build affine CVXPY expressions for ``A``, ``b``, ``c``, and ``d``.

        Linked parameter expressions should normally be supplied by original
        parameter ID. Unmapped parameters were frozen as constants before this
        affine map was extracted.
        """

        expressions = _normalise_expression_keys(parameter_expr_by_id)
        packed = self._parameter_vector_expression(expressions)
        affine = self._affine_map

        # CVXPY does not support sparse three-dimensional constants. Assemble
        # A entry-by-entry from sparse coefficient matrices, retaining a dense
        # symbolic matrix only at this final (typically small) lifted layer.
        rows: list[cp.Expression] = []
        for row in range(self.constraint_size):
            entries: list[cp.Expression] = []
            for column in range(self.canonical_size):
                entry: cp.Expression = cp.Constant(0.0)
                for parameter_index, coefficient in enumerate(affine.A):
                    value = float(coefficient[row, column])
                    if value:
                        entry = entry + value * packed[parameter_index]
                entries.append(entry)
            rows.append(cp.hstack(entries))
        A = cp.vstack(rows) if rows else cp.Constant(np.empty((0, self.canonical_size)))
        b = _symbolic_linear_combination(affine.b, packed)
        c = _symbolic_linear_combination(affine.c, packed)
        d = cp.sum(cp.multiply(affine.d, cp.hstack(packed)))
        return CanonicalExpressions(A=A, b=b, c=c, d=d)

    def recovery_expressions(self, u: cp.Expression) -> dict[int, cp.Expression]:
        """Return source-variable recovery expressions keyed by variable ID."""

        return self.recovery_map.expressions(u)

    def recover_numeric(self, u: ArrayLike) -> dict[int, NDArray[np.float64]]:
        """Recover source-variable values keyed by variable ID."""

        return self.recovery_map.numeric(u)

    def _parameter_vector_numeric(self, overrides: Mapping[int, ArrayLike]) -> NDArray[np.float64]:
        vector = np.zeros(len(self._affine_map.A), dtype=float)
        for spec in self.parameter_specs:
            if spec.parameter_id in overrides:
                value = overrides[spec.parameter_id]
            elif spec.mapped:
                expression = self.parameter_map[_parameter_by_id(self.lower_problem, spec.parameter_id)]
                value = expression.value
            else:
                value = _parameter_by_id(self.lower_problem, spec.parameter_id).value
            if value is None:
                raise ParameterMappingError(f"No numeric value is available for lower parameter {spec.name!r}.")
            vector[spec.offset : spec.offset + spec.internal_size] = spec.pack_numeric(value)
        vector[-1] = 1.0
        return vector

    def _parameter_vector_expression(
        self,
        expressions: Mapping[int, cp.Expression],
    ) -> list[cp.Expression]:
        vector: list[cp.Expression] = [cp.Constant(0.0) for _ in self._affine_map.A]
        for spec in self.parameter_specs:
            if spec.parameter_id in expressions:
                expression = expressions[spec.parameter_id]
            elif spec.mapped:
                expression = self.parameter_map[_parameter_by_id(self.lower_problem, spec.parameter_id)]
            else:
                value = _parameter_by_id(self.lower_problem, spec.parameter_id).value
                if value is None:
                    raise ParameterMappingError(f"Fixed lower parameter {spec.name!r} has no value.")
                expression = cp.Constant(value)
            packed = cp.reshape(spec.pack_expression(expression), (spec.internal_size,), order="F")
            for index in range(spec.internal_size):
                vector[spec.offset + index] = packed[index]
        vector[-1] = cp.Constant(1.0)
        return vector

    def _evaluate_affine_map(self, packed: NDArray[np.float64]) -> CanonicalData:
        affine = self._affine_map
        A = sp.csc_array((self.constraint_size, self.canonical_size), dtype=float)
        for value, coefficient in zip(packed, affine.A):
            if value:
                A = A + float(value) * coefficient
        return CanonicalData(
            A=A,
            b=affine.b @ packed,
            c=affine.c @ packed,
            d=float(affine.d @ packed),
        )


def validate_lower(
    lower_problem: cp.Problem,
    parameter_map: Mapping[cp.Parameter, cp.Expression],
) -> None:
    """Validate the source-level requirements of an SOCP lower model."""

    if not isinstance(lower_problem, cp.Problem):
        raise ValidationError("lower_problem must be a cvxpy.Problem.")
    if not isinstance(lower_problem.objective, cp.Minimize):
        raise ValidationError("The lower problem must be a minimization problem.")
    if lower_problem.objective.expr.is_complex():
        raise UnsupportedModelError("The lower objective must be real-valued.")
    if lower_problem.is_mixed_integer():
        raise UnsupportedModelError("Mixed-integer lower problems are not supported.")
    for variable in lower_problem.variables():
        if variable.is_complex():
            raise UnsupportedModelError(f"Complex lower variable {variable.name()!r} is not supported.")

    lower_parameters = {parameter.id: parameter for parameter in lower_problem.parameters()}
    seen: set[int] = set()
    try:
        items = tuple(parameter_map.items())
    except AttributeError as error:
        raise ParameterMappingError("parameter_map must be a mapping.") from error
    for parameter, linked_expression in items:
        if not isinstance(parameter, cp.Parameter):
            raise ParameterMappingError("Every parameter_map key must be a cvxpy.Parameter.")
        if parameter.id not in lower_parameters:
            raise ParameterMappingError(f"Mapped parameter {parameter.name()!r} does not occur in the lower problem.")
        if parameter.id in seen:
            raise ParameterMappingError(f"Parameter {parameter.name()!r} is mapped more than once.")
        seen.add(parameter.id)
        try:
            expression = cp.Expression.cast_to_const(linked_expression)
        except Exception as error:
            raise ParameterMappingError(
                f"Value linked to parameter {parameter.name()!r} is not a CVXPY expression."
            ) from error
        if expression.shape != parameter.shape:
            raise ParameterMappingError(
                f"Parameter {parameter.name()!r} has shape {parameter.shape}, but its linked "
                f"expression has shape {expression.shape}."
            )
        if expression.is_complex():
            raise ParameterMappingError(f"Expression linked to parameter {parameter.name()!r} must be real.")
        if not expression.is_affine():
            raise ParameterMappingError(f"Expression linked to parameter {parameter.name()!r} must be affine.")

    for parameter in lower_problem.parameters():
        if parameter.id not in seen and parameter.value is None:
            raise ParameterMappingError(f"Unmapped lower parameter {parameter.name()!r} must have a fixed value.")

    canonical_problem = _freeze_unmapped_parameters(lower_problem, seen)
    if not canonical_problem.is_dcp():
        raise ValidationError("The lower problem must satisfy CVXPY's DCP rules.")
    if not canonical_problem.is_dpp():
        raise ValidationError("The lower problem must satisfy CVXPY's DPP rules with respect to the mapped parameters.")

    _reject_approximate_source_nodes(lower_problem)


def canonicalize_lower(
    lower_problem: cp.Problem,
    parameter_map: Mapping[cp.Parameter, cp.Expression],
) -> CanonicalLowerProblem:
    """Canonicalize a supported lower DPP once using CVXPY and Clarabel."""

    validate_lower(lower_problem, parameter_map)
    mapping = dict(parameter_map)
    canonical_problem = _freeze_unmapped_parameters(lower_problem, {parameter.id for parameter in mapping})
    try:
        data, chain, inverse_data = canonical_problem.get_problem_data(
            cp.CLARABEL,
            enforce_dpp=True,
            solver_opts={"use_quad_obj": False},
        )
    except Exception as error:
        raise CanonicalizationError(
            "CVXPY could not canonicalize the lower problem for Clarabel with use_quad_obj=False."
        ) from error

    if not isinstance(data, Mapping):
        raise UnsupportedModelError(
            "The lower problem has no canonical optimization variable; "
            "constant-only lower problems are outside this release."
        )
    if "P" in data and data["P"] is not None and data["P"].nnz:
        raise CanonicalizationError("CVXPY retained a quadratic objective despite use_quad_obj=False.")
    param_prog = data.get("param_prob")
    if param_prog is None:
        raise CanonicalizationError("CVXPY did not expose its ParamConeProg.")
    try:
        cone_layout = ConeLayout.from_dims(data["dims"])
    except ValueError as error:
        raise UnsupportedConeError(str(error)) from error
    _audit_source_atoms(lower_problem)
    _audit_reduction_chain(chain)

    canonical_size = int(param_prog.x.size)
    constraint_size = int(data["A"].shape[0])
    if cone_layout.size != constraint_size:
        raise CanonicalizationError(
            f"Cone dimensions account for {cone_layout.size} rows, but A has {constraint_size}."
        )
    if data["A"].shape[1] != canonical_size:
        raise CanonicalizationError("CVXPY reported inconsistent canonical variable dimensions.")

    internal_mapping = {
        parameter: mapping[original]
        for original in lower_problem.parameters()
        if original in mapping
        for parameter in canonical_problem.parameters()
        if parameter.id == original.id
    }
    specs = _extract_parameter_specs(canonical_problem, internal_mapping, chain, param_prog)
    affine_map = _extract_affine_map(param_prog, constraint_size, canonical_size)
    recoveries = _extract_recovery_specs(canonical_problem, chain, inverse_data, param_prog, canonical_size)
    return CanonicalLowerProblem(
        lower_problem=lower_problem,
        canonical_problem=canonical_problem,
        parameter_map=internal_mapping,
        param_prog=param_prog,
        reduction_chain=tuple(chain.reductions),
        inverse_data=tuple(inverse_data),
        canonical_variable_offsets=param_prog.var_id_to_col,
        cone_layout=cone_layout,
        canonical_size=canonical_size,
        constraint_size=constraint_size,
        parameter_specs=specs,
        recovery_specs=recoveries,
        _affine_map=affine_map,
        fixed_parameter_values={
            parameter.id: np.asarray(parameter.value, dtype=float)
            for parameter in lower_problem.parameters()
            if parameter not in mapping
        },
    )


def _freeze_unmapped_parameters(problem: cp.Problem, mapped_ids: set[int]) -> cp.Problem:
    """Replace fixed-data parameters by constants before DPP canonicalization."""

    replacements = {
        parameter.id: cp.Constant(parameter.value)
        for parameter in problem.parameters()
        if parameter.id not in mapped_ids
    }
    if not replacements:
        return problem

    def replace(expression: cp.Expression) -> cp.Expression:
        if isinstance(expression, cp.Parameter) and expression.id in replacements:
            return replacements[expression.id]
        if not any(parameter.id in replacements for parameter in expression.parameters()):
            return expression
        new_args = [replace(argument) for argument in expression.args]
        return expression.copy(new_args)

    objective = type(problem.objective)(replace(problem.objective.expr))
    constraints: list[cp.Constraint] = []
    for constraint in problem.constraints:
        arguments = [replace(argument) for argument in constraint.args]
        data = constraint.get_data()
        constraints.append(type(constraint)(*(arguments + data)) if data is not None else type(constraint)(*arguments))
    return cp.Problem(objective, constraints)


def _reject_approximate_source_nodes(problem: cp.Problem) -> None:
    expressions = [problem.objective.expr]
    expressions.extend(argument for constraint in problem.constraints for argument in constraint.args)
    seen: set[int] = set()
    stack = list(expressions)
    while stack:
        expression = stack.pop()
        if id(expression) in seen:
            continue
        seen.add(id(expression))
        if isinstance(expression, (PowerApprox, PnormApprox, GeoMeanApprox)):
            error = float(getattr(expression, "approx_error", 0.0))
            if error > 10 * np.finfo(float).eps:
                raise ApproximateCanonicalizationError(
                    f"Atom {type(expression).__name__} uses a numerical approximation "
                    f"(error {error:.3g}); exact source atoms are required."
                )
        stack.extend(getattr(expression, "args", ()))
    for constraint in problem.constraints:
        if isinstance(constraint, (RelEntrConeQuad, OpRelEntrConeQuad)):
            raise ApproximateCanonicalizationError(
                f"Constraint {type(constraint).__name__} uses quadrature approximation."
            )


def _audit_source_atoms(problem: cp.Problem) -> None:
    """Enforce the explicit pointwise-graph atom allowlist for SOCP mode."""

    expressions = [problem.objective.expr]
    expressions.extend(argument for constraint in problem.constraints for argument in constraint.args)
    seen: set[int] = set()
    stack = list(expressions)
    while stack:
        expression = stack.pop()
        if id(expression) in seen:
            continue
        seen.add(id(expression))
        if isinstance(expression, Atom) and not isinstance(expression, AffAtom):
            atom_type = type(expression)
            if atom_type not in _AUDITED_NONLINEAR_ATOMS:
                raise UnsupportedModelError(
                    f"Atom {atom_type.__name__} is not in BLVpy's audited exact SOCP canonicalization allowlist."
                )
            if isinstance(expression, PnormApprox):
                p = float(expression.p)
                if p not in {1.0, 2.0} and not np.isinf(p):
                    raise UnsupportedModelError(
                        f"Atom {atom_type.__name__} with p={p:g} is outside the "
                        "audited LP/SOCP norm cases p in {1, 2, inf}."
                    )
            if isinstance(expression, PowerApprox):
                p = float(expression.p.value)
                if p not in {1.0, 2.0}:
                    raise UnsupportedModelError(
                        f"Atom {atom_type.__name__} with p={p:g} is outside the audited affine/quadratic power cases."
                    )
        stack.extend(getattr(expression, "args", ()))


def _audit_reduction_chain(chain: Any) -> None:
    names = tuple(type(reduction).__name__ for reduction in chain.reductions)
    if names != _AUDITED_REDUCTION_CHAIN and names != (
        "Dcp2Cone",
        "CvxAttr2Constr",
        "ExactCone2Cone",
        "EliminateZeroSized",
        "ConeMatrixStuffing",
        "CLARABEL",
    ):
        raise CanonicalizationError("CVXPY selected an unaudited canonicalization chain: " + " -> ".join(names))


def _extract_parameter_specs(
    problem: cp.Problem,
    mapping: Mapping[cp.Parameter, cp.Expression],
    chain: Any,
    param_prog: Any,
) -> tuple[ParameterSpec, ...]:
    mapped_ids = {parameter.id for parameter in mapping}
    internal_by_original: dict[int, tuple[cp.Parameter, ParameterTransform]] = {}
    attr_reduction = next(
        (reduction for reduction in chain.reductions if type(reduction).__name__ == "CvxAttr2Constr"),
        None,
    )
    replaced = getattr(attr_reduction, "_parameters", {}) if attr_reduction is not None else {}
    for parameter in problem.parameters():
        internal = replaced.get(parameter, parameter)
        internal_by_original[parameter.id] = (internal, _parameter_transform(parameter, internal))

    specs: list[ParameterSpec] = []
    for parameter in problem.parameters():
        internal, transform = internal_by_original[parameter.id]
        if internal.id not in param_prog.param_id_to_col:
            raise CanonicalizationError(f"CVXPY's canonical parameter vector omits {parameter.name()!r}.")
        sparse_indices: tuple[tuple[int, ...], ...] = ()
        if transform == "sparse":
            sparse_indices = tuple(tuple(int(value) for value in axis) for axis in parameter.sparse_idx)
        specs.append(
            ParameterSpec(
                parameter_id=int(parameter.id),
                name=parameter.name() or f"param_{parameter.id}",
                shape=tuple(int(size) for size in parameter.shape),
                size=int(parameter.size),
                mapped=parameter.id in mapped_ids,
                internal_parameter_id=int(internal.id),
                internal_shape=tuple(int(size) for size in internal.shape),
                internal_size=int(internal.size),
                offset=int(param_prog.param_id_to_col[internal.id]),
                transform=transform,
                sparse_indices=sparse_indices,
            )
        )
    return tuple(specs)


def _parameter_transform(parameter: cp.Parameter, internal: cp.Parameter) -> ParameterTransform:
    if parameter is internal:
        return "identity"
    attributes = parameter.attributes
    if any(attributes.get(name, False) for name in ("symmetric", "PSD", "NSD")):
        return "symmetric"
    if attributes.get("diag", False):
        return "diagonal"
    if attributes.get("sparsity", False):
        return "sparse"
    raise CanonicalizationError(f"Unsupported parameter dimension reduction for {parameter.name()!r}.")


def _extract_affine_map(param_prog: Any, rows: int, columns: int) -> _DataAffineMap:
    parameter_vector_size = int(param_prog.total_param_size) + 1
    A_coefficients: list[sp.csc_array] = []
    b_coefficients = np.empty((rows, parameter_vector_size), dtype=float)
    c_coefficients = np.empty((columns, parameter_vector_size), dtype=float)
    d_coefficients = np.empty(parameter_vector_size, dtype=float)
    param_prog.reduced_A.cache(True)
    for index in range(parameter_vector_size):
        basis = np.zeros(parameter_vector_size, dtype=float)
        basis[index] = 1.0
        c_sparse, d = _matrix_and_offset(param_prog.q, basis, columns)
        A, b = param_prog.reduced_A.get_matrix_from_tensor(basis, with_offset=True)
        # ConeMatrixStuffing stores the affine constraint expression F @ u + g
        # while conic solver interfaces expose ``A=-F`` and ``b=g``.  BLVpy's
        # public convention follows the latter: A @ u + s == b.
        A_coefficients.append(-sp.csc_array(A, dtype=float))
        b_coefficients[:, index] = np.asarray(b, dtype=float).reshape(-1)
        c_coefficients[:, index] = np.asarray(c_sparse.toarray(), dtype=float).reshape(-1)
        d_coefficients[index] = float(np.asarray(d).reshape(()))
    for array in (b_coefficients, c_coefficients, d_coefficients):
        array.setflags(write=False)
    return _DataAffineMap(
        A=tuple(A_coefficients),
        b=b_coefficients,
        c=c_coefficients,
        d=d_coefficients,
    )


def _matrix_and_offset(tensor: Any, parameter_vector: NDArray[np.float64], length: int) -> tuple[Any, Any]:
    # This is CVXPY's stable tensor contract used by ParamConeProg itself.
    from cvxpy.cvxcore.python import canonInterface

    return canonInterface.get_matrix_from_tensor(tensor, parameter_vector, length, with_offset=True)


def _extract_recovery_specs(
    problem: cp.Problem,
    chain: Any,
    inverse_data: list[Any],
    param_prog: Any,
    canonical_size: int,
) -> tuple[RecoverySpec, ...]:
    zero = _recover_source_values(np.zeros(canonical_size), problem, chain, inverse_data, param_prog)
    matrices = {variable.id: np.empty((variable.size, canonical_size), dtype=float) for variable in problem.variables()}
    for column in range(canonical_size):
        basis = np.zeros(canonical_size)
        basis[column] = 1.0
        recovered = _recover_source_values(basis, problem, chain, inverse_data, param_prog)
        for variable in problem.variables():
            matrices[variable.id][:, column] = np.asarray(recovered[variable.id]).reshape(-1, order="F") - np.asarray(
                zero[variable.id]
            ).reshape(-1, order="F")

    specs: list[RecoverySpec] = []
    for variable in problem.variables():
        if variable.id not in zero:
            raise CanonicalizationError(
                f"CVXPY did not provide recovery metadata for lower variable {variable.name()!r}."
            )
        specs.append(
            RecoverySpec(
                variable_id=int(variable.id),
                name=variable.name() or f"var_{variable.id}",
                shape=tuple(int(size) for size in variable.shape),
                matrix=matrices[variable.id],
                offset=np.asarray(zero[variable.id]).reshape(-1, order="F"),
            )
        )
    return tuple(specs)


def _recover_source_values(
    canonical_value: NDArray[np.float64],
    problem: cp.Problem,
    chain: Any,
    inverse_data: list[Any],
    param_prog: Any,
) -> dict[int, NDArray[np.float64]]:
    dual_values = {constraint.id: np.zeros(constraint.shape, dtype=float) for constraint in param_prog.constraints}
    solution = Solution(
        cvxpy_settings.OPTIMAL,
        0.0,
        {param_prog.x.id: canonical_value},
        dual_values,
        {},
    )
    try:
        # Skip the solver interface. Its invert step expects a Clarabel-native
        # object and has no role in the source-variable affine map.
        for reduction, inverse in reversed(list(zip(chain.reductions[:-1], inverse_data[:-1]))):
            # CvxAttr2Constr.invert projects ordinary sign/bound-attributed
            # variables. Projection is harmless for a feasible solver point
            # but nonlinear, so it cannot define the fixed affine recovery map
            # required by DBP. Its public ``var_forward`` performs exactly the
            # desired linear unpacking, including symmetric/sparse variables.
            if type(reduction).__name__ == "CvxAttr2Constr":
                solution = Solution(
                    solution.status,
                    solution.opt_val,
                    reduction.var_forward(solution.primal_vars),
                    solution.dual_vars,
                    solution.attr,
                )
            else:
                solution = reduction.invert(solution, inverse)
    except Exception as error:
        raise CanonicalizationError("CVXPY source-variable recovery failed.") from error
    result: dict[int, NDArray[np.float64]] = {}
    for variable in problem.variables():
        value = solution.primal_vars.get(variable.id)
        if value is not None:
            result[variable.id] = np.asarray(value, dtype=float)
    return result


def _normalise_value_keys(
    values: Mapping[cp.Parameter | int, ArrayLike],
) -> dict[int, ArrayLike]:
    result: dict[int, ArrayLike] = {}
    for key, value in values.items():
        parameter_id = int(key.id) if isinstance(key, cp.Parameter) else int(key)
        result[parameter_id] = value
    return result


def _normalise_expression_keys(
    values: Mapping[cp.Parameter | int, cp.Expression],
) -> dict[int, cp.Expression]:
    result: dict[int, cp.Expression] = {}
    for key, value in values.items():
        parameter_id = int(key.id) if isinstance(key, cp.Parameter) else int(key)
        result[parameter_id] = cp.Expression.cast_to_const(value)
    return result


def _parameter_by_id(problem: cp.Problem, parameter_id: int) -> cp.Parameter:
    for parameter in problem.parameters():
        if parameter.id == parameter_id:
            return parameter
    raise CanonicalizationError(f"Unknown lower parameter ID {parameter_id}.")


def _symbolic_linear_combination(
    coefficients: NDArray[np.float64],
    parameters: list[cp.Expression],
) -> cp.Expression:
    rows = [
        cp.sum(cp.hstack([float(value) * parameter for value, parameter in zip(row, parameters)]))
        for row in coefficients
    ]
    return cp.hstack(rows) if rows else cp.Constant(np.empty(0))


def _readonly_vector(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float).reshape(-1).copy()
    array.setflags(write=False)
    return array
