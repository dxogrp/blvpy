"""Lossless DPP-to-conic canonicalization for lower-level problems.

This module deliberately stops at CVXPY's pre-solver conic representation.
The returned matrices use the solver convention ``A @ u + s == b``; no
Clarabel scaling or presolve data is involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from ._canonicalization.affine import (
    _DataAffineMap,
    _extract_affine_map,
    _readonly_vector,
    _symbolic_matrix_combination,
    _symbolic_vector_combination,
)
from ._canonicalization.affine import (
    _matrix_and_offset as _matrix_and_offset,
)
from ._canonicalization.audit import (
    _AUDITED_NONLINEAR_ATOMS as _AUDITED_NONLINEAR_ATOMS,
)
from ._canonicalization.audit import (
    _AUDITED_REDUCTION_CHAIN as _AUDITED_REDUCTION_CHAIN,
)
from ._canonicalization.audit import (
    _approximation_error as _approximation_error,
)
from ._canonicalization.audit import (
    _approximation_metadata as _approximation_metadata,
)
from ._canonicalization.audit import (
    _audit_reduction_chain,
    _audit_source_atoms,
    _validate_lower,
)
from ._canonicalization.audit import (
    _reject_approximate_source_nodes as _reject_approximate_source_nodes,
)
from ._canonicalization.audit import (
    _safe_metadata_repr as _safe_metadata_repr,
)
from ._canonicalization.parameters import (
    ParameterTransform,
    _extract_parameter_specs,
    _freeze_unmapped_parameters,
    _normalise_expression_keys,
    _normalise_value_keys,
    _parameter_by_id,
)
from ._canonicalization.parameters import (
    _parameter_transform as _parameter_transform,
)
from ._canonicalization.recovery import _extract_recovery_specs
from ._canonicalization.recovery import (
    _recover_source_values as _recover_source_values,
)
from .cones import ConeLayout
from .errors import (
    ApproximateCanonicalizationError as ApproximateCanonicalizationError,
)
from .errors import (
    CanonicalizationError,
    ParameterMappingError,
    UnsupportedConeError,
    UnsupportedModelError,
)
from .errors import ValidationError as ValidationError


@dataclass(frozen=True, slots=True)
class CanonicalData:
    """Numerical data for one evaluated canonical lower problem.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        Constraint matrix in the convention ``A @ u + s == b``.
    b : array-like
        One-dimensional right-hand-side vector.
    c : array-like
        One-dimensional linear-objective vector.
    d : float
        Scalar objective offset, so the primal objective is ``c.T @ u + d``.

    Raises
    ------
    ValueError
        If dimensions are inconsistent or any data are nonfinite.

    Notes
    -----
    ``b`` and ``c`` are stored as read-only ``float64`` arrays. The row order
    of ``A`` and ``b`` is described by the corresponding
    :class:`blvpy.ConeLayout`.
    """

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
    """Affine recovery maps for the original lower variables.

    Parameters
    ----------
    specs : tuple of RecoverySpec
        Per-variable maps in the original lower problem's variable order.
    """

    specs: tuple[RecoverySpec, ...]

    def expressions(self, u: cp.Expression) -> dict[int, cp.Expression]:
        """Construct source-variable recovery expressions.

        Parameters
        ----------
        u : cvxpy.Expression
            Canonical primal vector.

        Returns
        -------
        dict[int, cvxpy.Expression]
            Shaped affine expressions keyed by original CVXPY variable ID.
        """

        return {spec.variable_id: spec.expression(u) for spec in self.specs}

    def numeric(self, u: ArrayLike) -> dict[int, NDArray[np.float64]]:
        """Evaluate all source-variable recovery maps.

        Parameters
        ----------
        u : array-like
            Canonical primal vector.

        Returns
        -------
        dict[int, numpy.ndarray]
            Shaped values keyed by original CVXPY variable ID.

        Raises
        ------
        ValueError
            If ``u`` has an incompatible length.
        """

        return {spec.variable_id: spec.numeric(u) for spec in self.specs}


@dataclass(frozen=True, slots=True)
class CanonicalExpressions:
    """Symbolic affine data of a canonical lower problem.

    Parameters
    ----------
    A : cvxpy.Expression
        Canonical constraint matrix as an affine expression of linked upper
        variables.
    b : cvxpy.Expression
        Canonical right-hand-side vector.
    c : cvxpy.Expression
        Canonical linear-objective vector.
    d : cvxpy.Expression
        Canonical scalar objective offset.

    Notes
    -----
    These expressions use the convention ``A @ u + s == b`` and objective
    ``c.T @ u + d``. They are primarily intended for advanced inspection;
    BLVPY constructs the lifted model from them internally.
    """

    A: cp.Expression
    b: cp.Expression
    c: cp.Expression
    d: cp.Expression


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Description of one parameter in CVXPY's packed canonical data.

    Parameters
    ----------
    parameter_id : int
        ID of the parameter before CVXPY attribute reduction.
    name : str
        Source parameter name used in diagnostics.
    shape : tuple of int
        Original parameter shape.
    size : int
        Number of entries in the original shape.
    mapped : bool
        Whether the parameter represents a linked upper variable.
    internal_parameter_id : int
        ID after CVXPY attribute reduction.
    internal_shape : tuple of int
        Shape after attribute reduction.
    internal_size : int
        Number of packed entries after attribute reduction.
    offset : int
        Starting column in CVXPY's packed parameter vector.
    transform : {"identity", "symmetric", "diagonal", "sparse"}, optional
        Transformation from the source value to the packed representation.
    sparse_indices : tuple of tuple of int, optional
        Source indices retained by the ``"sparse"`` transformation.

    Notes
    -----
    This is provisional inspection metadata returned through
    :class:`blvpy.CanonicalLowerProblem`; users normally do not construct it.
    """

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
        """Pack a numeric source value in CVXPY's internal order.

        Parameters
        ----------
        value : array-like
            Finite value whose shape exactly matches ``shape``.

        Returns
        -------
        numpy.ndarray
            One-dimensional packed value of length ``internal_size``.

        Raises
        ------
        ParameterMappingError
            If the shape is wrong or any entry is nonfinite.
        """

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
        """Pack an affine CVXPY expression in the internal parameter order.

        Parameters
        ----------
        value : cvxpy.Expression or array-like
            Affine expression with shape ``shape``.

        Returns
        -------
        cvxpy.Expression
            One-dimensional packed expression of length
            ``internal_size``.

        Raises
        ------
        ParameterMappingError
            If the expression has the wrong shape or is not affine.
        """

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
    """Fixed affine recovery of one source lower variable.

    Parameters
    ----------
    variable_id : int
        ID of the original CVXPY lower variable.
    name : str
        Original variable name used in diagnostics.
    shape : tuple of int
        Original variable shape.
    matrix : array-like
        Two-dimensional matrix multiplying the canonical primal vector.
    offset : array-like
        Vector added before reshaping in Fortran order to ``shape``.

    Raises
    ------
    ValueError
        If ``matrix`` and ``offset`` do not describe the requested source
        shape.

    Notes
    -----
    Recovery has the form ``reshape(matrix @ u + offset, shape, order="F")``.
    """

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
        """Construct the shaped affine recovery expression.

        Parameters
        ----------
        u : cvxpy.Expression
            Canonical primal vector.

        Returns
        -------
        cvxpy.Expression
            Source-shaped affine expression.
        """

        vector = cp.reshape(cp.Expression.cast_to_const(u), (self.matrix.shape[1],), order="F")
        flat = self.matrix @ vector + self.offset
        return cp.reshape(flat, self.shape, order="F")

    def numeric(self, u: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the shaped affine recovery map.

        Parameters
        ----------
        u : array-like
            Canonical primal vector.

        Returns
        -------
        numpy.ndarray
            Recovered value with ``shape``.

        Raises
        ------
        ValueError
            If ``u`` has an incompatible length.
        """

        vector = np.asarray(u, dtype=float).reshape(-1)
        if vector.size != self.matrix.shape[1]:
            raise ValueError(f"Canonical vector has length {vector.size}; expected {self.matrix.shape[1]}.")
        return np.asarray(self.matrix @ vector + self.offset).reshape(self.shape, order="F")


@dataclass(frozen=True, slots=True)
class CanonicalLowerProblem:
    """Fixed exact conic canonicalization of a lower problem.

    Instances are produced and cached by
    :meth:`blvpy.BilevelProblem.canonicalize`.
    They expose the affine canonical data and source-recovery metadata for
    advanced numerical inspection; direct construction is not a supported
    modeling workflow.

    Attributes
    ----------
    canonical_variable_offsets : collections.abc.Mapping[int, int]
        Read-only mapping from each CVXPY canonical variable ID to its
        starting canonical column.
    cone_layout : ConeLayout
        Ordered zero, nonnegative, second-order, exponential, and 3D
        power-cone blocks.
    canonical_size : int
        Length of the canonical primal vector ``u``.
    constraint_size : int
        Length of the canonical slack and dual vectors.
    parameter_specs : tuple of ParameterSpec
        Packing metadata for parameters retained in the affine data map.
    recovery_specs : tuple of RecoverySpec
        Affine source-variable recovery maps.
    fixed_parameter_values : collections.abc.Mapping[int, numpy.ndarray]
        Read-only mapping of unmapped parameter IDs to read-only value
        snapshots captured at canonicalization time.

    Notes
    -----
    The represented program is ``min c(x).T @ u + d(x)`` subject to
    ``A(x) @ u + s == b(x)`` and ``s`` in
    :attr:`blvpy.CanonicalLowerProblem.cone_layout`. Its matrix
    convention is pre-solver CVXPY canonical data, before Clarabel scaling or
    presolve.
    """

    _source_problem: cp.Problem
    _canonical_problem: cp.Problem
    _parameter_links: Mapping[cp.Parameter, cp.Expression]
    canonical_variable_offsets: Mapping[int, int]
    cone_layout: ConeLayout
    canonical_size: int
    constraint_size: int
    parameter_specs: tuple[ParameterSpec, ...]
    recovery_specs: tuple[RecoverySpec, ...]
    _affine_map: _DataAffineMap
    fixed_parameter_values: Mapping[int, NDArray[np.float64]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_parameter_links",
            MappingProxyType(dict(self._parameter_links)),
        )
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
        """tuple of int: Original lower-variable IDs in CVXPY problem order."""

        return tuple(spec.variable_id for spec in self.recovery_specs)

    @property
    def recovery_map(self) -> AffineRecoveryMap:
        """AffineRecoveryMap: Combined recovery map for all source lower variables."""

        return AffineRecoveryMap(self.recovery_specs)

    @property
    def parameter_ids(self) -> tuple[int, ...]:
        """tuple of int: Retained parameter IDs in CVXPY problem order.

        Unmapped parameters are frozen into constant canonical data at
        canonicalization time and therefore do not appear here.
        """

        return tuple(spec.parameter_id for spec in self.parameter_specs)

    def apply_numeric(
        self,
        values: Mapping[cp.Parameter | int, ArrayLike] | None = None,
    ) -> CanonicalData:
        """Evaluate the affine canonical data numerically.

        Parameters
        ----------
        values : mapping or None, optional
            Parameter overrides keyed by a retained CVXPY parameter object or
            its integer ID. Missing mapped values are read from their linked
            upper expressions. Unmapped parameters remain frozen at their
            canonicalization-time values.

        Returns
        -------
        CanonicalData
            Evaluated ``A``, ``b``, ``c``, and ``d``.

        Raises
        ------
        ParameterMappingError
            If a required linked value is absent, has the wrong shape, or is
            nonfinite.
        ValueError
            If the evaluated canonical dimensions or values are invalid.
        """

        overrides = _normalise_value_keys(values or {})
        packed = self._parameter_vector_numeric(overrides)
        return self._evaluate_affine_map(packed)

    def build_data_expressions(
        self,
        parameter_expr_by_id: Mapping[cp.Parameter | int, cp.Expression],
    ) -> CanonicalExpressions:
        """Build symbolic affine expressions for the canonical data.

        Parameters
        ----------
        parameter_expr_by_id : mapping
            Optional replacements keyed by a retained CVXPY parameter object
            or its integer ID. Values must be affine CVXPY expressions with
            the source parameter shape. Missing mapped entries use their
            linked upper expressions.

        Returns
        -------
        CanonicalExpressions
            Affine expressions for ``A``, ``b``, ``c``, and ``d``.

        Raises
        ------
        ParameterMappingError
            If a replacement has an incompatible shape or is not affine, or
            if no value is available for a required parameter.

        Notes
        -----
        Unmapped parameters were frozen as constants before the affine map was
        extracted.
        """

        expressions = _normalise_expression_keys(parameter_expr_by_id)
        packed = self._parameter_vector_expression(expressions)
        packed_vector = cp.hstack(packed)
        affine = self._affine_map

        # CVXPY supports sparse two-dimensional constants but not sparse 3-D
        # tensors. Flatten the coefficient matrices into one sparse linear
        # operator, apply it to the packed parameter vector, and reshape once.
        # This avoids an expression node for every entry of every coefficient.
        A = _symbolic_matrix_combination(
            affine.A,
            packed_vector,
            self.constraint_size,
            self.canonical_size,
        )
        b = _symbolic_vector_combination(affine.b, packed_vector)
        c = _symbolic_vector_combination(affine.c, packed_vector)
        d = cp.Constant(affine.d) @ packed_vector
        return CanonicalExpressions(A=A, b=b, c=c, d=d)

    def recovery_expressions(self, u: cp.Expression) -> dict[int, cp.Expression]:
        """Construct recovery expressions for all source lower variables.

        Parameters
        ----------
        u : cvxpy.Expression
            Canonical primal vector.

        Returns
        -------
        dict[int, cvxpy.Expression]
            Source-shaped expressions keyed by original variable ID.
        """

        return self.recovery_map.expressions(u)

    def recover_numeric(self, u: ArrayLike) -> dict[int, NDArray[np.float64]]:
        """Recover all source lower variables from a canonical vector.

        Parameters
        ----------
        u : array-like
            Canonical primal vector.

        Returns
        -------
        dict[int, numpy.ndarray]
            Source-shaped values keyed by original variable ID.

        Raises
        ------
        ValueError
            If the canonical vector has an incompatible length.
        """

        return self.recovery_map.numeric(u)

    def _parameter_vector_numeric(self, overrides: Mapping[int, ArrayLike]) -> NDArray[np.float64]:
        vector = np.zeros(len(self._affine_map.A), dtype=float)
        for spec in self.parameter_specs:
            if spec.parameter_id in overrides:
                value = overrides[spec.parameter_id]
            elif spec.mapped:
                expression = self._parameter_links[_parameter_by_id(self._source_problem, spec.parameter_id)]
                value = expression.value
            else:
                value = _parameter_by_id(self._source_problem, spec.parameter_id).value
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
                expression = self._parameter_links[_parameter_by_id(self._source_problem, spec.parameter_id)]
            else:
                value = _parameter_by_id(self._source_problem, spec.parameter_id).value
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


def _canonicalize_lower(
    lower_problem: cp.Problem,
    parameter_links: Mapping[cp.Parameter, cp.Expression],
) -> CanonicalLowerProblem:
    """Canonicalize a supported lower DPP once using CVXPY and Clarabel."""

    _validate_lower(lower_problem, parameter_links)
    mapping = dict(parameter_links)
    canonical_problem = _freeze_unmapped_parameters(lower_problem, {parameter.id for parameter in mapping})
    if isinstance(canonical_problem.objective, cp.Maximize):
        canonical_problem = cp.Problem(
            cp.Minimize(-canonical_problem.objective.expr),
            canonical_problem.constraints,
        )
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
    specs = _extract_parameter_specs(
        canonical_problem,
        internal_mapping,
        chain,
        param_prog,
        parameter_spec_factory=ParameterSpec,
    )
    affine_map = _extract_affine_map(param_prog, constraint_size, canonical_size)
    recoveries = _extract_recovery_specs(
        canonical_problem,
        chain,
        inverse_data,
        param_prog,
        canonical_size,
        recovery_spec_factory=RecoverySpec,
    )
    return CanonicalLowerProblem(
        _source_problem=lower_problem,
        _canonical_problem=canonical_problem,
        _parameter_links=internal_mapping,
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
