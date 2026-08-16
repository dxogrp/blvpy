"""Product-cone utilities for BLVPY's SOCP canonical form.

CVXPY orders conic rows as zero, nonnegative, and then second-order
cone blocks.  :class:`ConeLayout` records that order once and uses it for
both symbolic membership constraints and numerical diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Any, Literal

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

ConeKind = Literal["zero", "nonnegative", "second_order"]


@dataclass(frozen=True, slots=True)
class ConeBlock:
    """One contiguous block in a canonical product-cone vector.

    Parameters
    ----------
    kind : {"zero", "nonnegative", "second_order"}
        Cone represented by the block.
    start : int
        Inclusive zero-based row offset.
    stop : int
        Exclusive row offset; it must be greater than ``start``.
    index : int, default=0
        Zero-based index among blocks of the same kind. It distinguishes
        multiple second-order cones.

    Raises
    ------
    ValueError
        If the kind, offsets, or index are invalid.
    """

    kind: ConeKind
    start: int
    stop: int
    index: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"zero", "nonnegative", "second_order"}:
            raise ValueError(f"Unknown cone kind {self.kind!r}.")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("A cone block must be a nonempty forward slice.")
        if self.index < 0:
            raise ValueError("A cone block index must be nonnegative.")

    @property
    def size(self) -> int:
        """int: Number of scalar rows in the block."""

        return self.stop - self.start

    @property
    def slice(self) -> slice:
        """slice: Python slice selecting the block from a canonical vector."""

        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class ConeLayout:
    """Ordered layout of a zero/nonnegative/SOC product cone.

    Parameters
    ----------
    zero : int, default=0
        Number of scalar rows in the zero-cone block.
    nonnegative : int, default=0
        Number of scalar rows in the nonnegative-cone block.
    second_order : tuple of int, optional
        Dimensions of the second-order cone blocks. Each dimension includes
        the scalar head and must be at least two.

    Raises
    ------
    ValueError
        If a dimension is negative, nonintegral, or an SOC dimension is less
        than two.

    Notes
    -----
    Rows follow CVXPY's canonical order: zero, nonnegative, then each SOC in
    sequence. The associated dual cone is unrestricted on zero-cone rows and
    self-dual on nonnegative and SOC rows.
    """

    zero: int = 0
    nonnegative: int = 0
    second_order: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        zero = _dimension(self.zero, "zero")
        nonnegative = _dimension(self.nonnegative, "nonnegative")
        try:
            second_order = tuple(
                _dimension(size, f"second_order[{position}]") for position, size in enumerate(self.second_order)
            )
        except TypeError as error:
            raise ValueError("second_order must be a sequence of cone sizes.") from error
        if any(size < 2 for size in second_order):
            raise ValueError("Every second-order cone must have dimension at least 2.")
        object.__setattr__(self, "zero", zero)
        object.__setattr__(self, "nonnegative", nonnegative)
        object.__setattr__(self, "second_order", second_order)

    @classmethod
    def from_dims(cls, dims: object) -> ConeLayout:
        """Build a layout from CVXPY cone dimensions.

        Parameters
        ----------
        dims : object or mapping
            A CVXPY ``ConeDims``-like object or mapping. CVXPY aliases such as
            ``f``, ``l``, and ``q`` are recognized.

        Returns
        -------
        ConeLayout
            Validated zero/nonnegative/SOC row layout.

        Raises
        ------
        ValueError
            If ``dims`` is ``None``, contains invalid dimensions, or declares
            nonempty PSD, exponential, or power cones.
        """

        if dims is None:
            raise ValueError("Cone dimensions cannot be None.")

        zero = _dim_value(dims, ("zero", "f"), 0)
        nonnegative = _dim_value(dims, ("nonnegative", "nonneg", "l"), 0)
        second_order = _dim_value(dims, ("second_order", "soc", "q"), ())

        unsupported: list[str] = []
        for label, names in (
            ("exponential", ("exp", "ep")),
            ("positive-semidefinite", ("psd", "s")),
            ("3D power", ("p3d", "p3")),
            ("N-dimensional power", ("pnd",)),
        ):
            value = _dim_value(dims, names, 0)
            if _has_cones(value):
                unsupported.append(label)
        if unsupported:
            rendered = ", ".join(unsupported)
            raise ValueError(f"Unsupported cone dimensions for SOCP mode: {rendered}.")

        return cls(
            zero=zero,
            nonnegative=nonnegative,
            second_order=tuple(second_order or ()),
        )

    @property
    def nonneg(self) -> int:
        """int: CVXPY-compatible alias for ``nonnegative``."""

        return self.nonnegative

    @property
    def soc(self) -> tuple[int, ...]:
        """tuple of int: CVXPY-compatible alias for ``second_order``."""

        return self.second_order

    @property
    def size(self) -> int:
        """int: Total number of scalar product-cone rows."""

        return self.zero + self.nonnegative + sum(self.second_order)

    @property
    def zero_slice(self) -> slice:
        """slice: Zero-cone rows, possibly an empty slice."""

        return slice(0, self.zero)

    @property
    def nonnegative_slice(self) -> slice:
        """slice: Nonnegative-cone rows, possibly an empty slice."""

        return slice(self.zero, self.zero + self.nonnegative)

    @property
    def nonneg_slice(self) -> slice:
        """slice: Alias for ``nonnegative_slice``."""

        return self.nonnegative_slice

    @property
    def second_order_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Ordered second-order cone row slices."""

        start = self.zero + self.nonnegative
        slices: list[slice] = []
        for size in self.second_order:
            slices.append(slice(start, start + size))
            start += size
        return tuple(slices)

    @property
    def soc_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Alias for ``second_order_slices``."""

        return self.second_order_slices

    @property
    def blocks(self) -> tuple[ConeBlock, ...]:
        """tuple of ConeBlock: All nonempty blocks in canonical row order."""

        blocks: list[ConeBlock] = []
        if self.zero:
            blocks.append(ConeBlock("zero", 0, self.zero))
        if self.nonnegative:
            blocks.append(
                ConeBlock(
                    "nonnegative",
                    self.nonnegative_slice.start,
                    self.nonnegative_slice.stop,
                )
            )
        blocks.extend(
            ConeBlock("second_order", block.start, block.stop, position)
            for position, block in enumerate(self.second_order_slices)
        )
        return tuple(blocks)

    def primal_constraints(self, value: cp.Expression | ArrayLike) -> tuple[cp.Constraint, ...]:
        """Construct CVXPY constraints for primal-cone membership.

        Parameters
        ----------
        value : cvxpy.Expression or array-like
            Real vector with :attr:`size` entries.

        Returns
        -------
        tuple of cvxpy.Constraint
            Zero equalities, nonnegative inequalities, and scalar-form SOC
            inequalities in canonical block order.

        Raises
        ------
        ValueError
            If ``value`` is complex or has the wrong number of entries.
        """

        vector = _expression_vector(value, self.size)
        constraints: list[cp.Constraint] = []
        if self.zero:
            constraints.append(vector[self.zero_slice] == 0)
        if self.nonnegative:
            constraints.append(vector[self.nonnegative_slice] >= 0)
        constraints.extend(_soc_constraint(vector, block) for block in self.second_order_slices)
        return tuple(constraints)

    def dual_constraints(self, value: cp.Expression | ArrayLike) -> tuple[cp.Constraint, ...]:
        """Construct CVXPY constraints for dual-cone membership.

        Parameters
        ----------
        value : cvxpy.Expression or array-like
            Real vector with :attr:`size` entries.

        Returns
        -------
        tuple of cvxpy.Constraint
            Nonnegative and SOC membership constraints. Zero-cone dual rows
            are unrestricted and therefore add no constraints.

        Raises
        ------
        ValueError
            If ``value`` is complex or has the wrong number of entries.
        """

        vector = _expression_vector(value, self.size)
        constraints: list[cp.Constraint] = []
        if self.nonnegative:
            constraints.append(vector[self.nonnegative_slice] >= 0)
        constraints.extend(_soc_constraint(vector, block) for block in self.second_order_slices)
        return tuple(constraints)

    def primal_distance(self, value: ArrayLike) -> float:
        """Compute distance to the primal product cone.

        Parameters
        ----------
        value : array-like
            Real numeric vector with :attr:`size` entries.

        Returns
        -------
        float
            Euclidean product-cone distance. With finite zero-cone entries,
            nonfinite entries in a nonnegative or second-order block produce
            positive infinity; NaN in a zero-cone block propagates to the
            result.

        Raises
        ------
        ValueError
            If ``value`` is complex, nonnumeric, or has the wrong size.
        """

        vector = _numeric_vector(value, self.size)
        squared_distance = float(np.dot(vector[self.zero_slice], vector[self.zero_slice]))
        squared_distance += _nonnegative_squared_distance(vector[self.nonnegative_slice])
        squared_distance += sum(_soc_squared_distance(vector[block]) for block in self.second_order_slices)
        return float(np.sqrt(squared_distance))

    def dual_distance(self, value: ArrayLike) -> float:
        """Compute distance to the dual product cone.

        Parameters
        ----------
        value : array-like
            Real numeric vector with :attr:`size` entries.

        Returns
        -------
        float
            Euclidean distance, with zero-cone dual rows unrestricted, or
            positive infinity for nonfinite constrained entries.

        Raises
        ------
        ValueError
            If ``value`` is complex, nonnumeric, or has the wrong size.
        """

        vector = _numeric_vector(value, self.size)
        squared_distance = _nonnegative_squared_distance(vector[self.nonnegative_slice])
        squared_distance += sum(_soc_squared_distance(vector[block]) for block in self.second_order_slices)
        return float(np.sqrt(squared_distance))

    def complementarity(self, primal: ArrayLike, dual: ArrayLike) -> float:
        """Compute the canonical primal-dual pairing.

        Parameters
        ----------
        primal : array-like
            Primal cone vector with :attr:`size` entries.
        dual : array-like
            Dual cone vector with :attr:`size` entries.

        Returns
        -------
        float
            Euclidean pairing ``primal @ dual``.

        Raises
        ------
        ValueError
            If either vector is complex, nonnumeric, or has the wrong size.
        """

        primal_vector = _numeric_vector(primal, self.size)
        dual_vector = _numeric_vector(dual, self.size)
        return float(primal_vector @ dual_vector)


def primal_cone_constraints(
    value: cp.Expression | ArrayLike,
    layout: ConeLayout,
) -> tuple[cp.Constraint, ...]:
    """Functional form of :meth:`ConeLayout.primal_constraints`."""

    return layout.primal_constraints(value)


def dual_cone_constraints(
    value: cp.Expression | ArrayLike,
    layout: ConeLayout,
) -> tuple[cp.Constraint, ...]:
    """Functional form of :meth:`ConeLayout.dual_constraints`."""

    return layout.dual_constraints(value)


def primal_cone_distance(value: ArrayLike, layout: ConeLayout) -> float:
    """Functional form of :meth:`ConeLayout.primal_distance`."""

    return layout.primal_distance(value)


def dual_cone_distance(value: ArrayLike, layout: ConeLayout) -> float:
    """Functional form of :meth:`ConeLayout.dual_distance`."""

    return layout.dual_distance(value)


def soc_distance(value: ArrayLike) -> float:
    """Euclidean distance to one second-order cone."""

    vector = _numeric_vector_unknown_size(value)
    if vector.size < 2:
        raise ValueError("A second-order cone vector must have at least two entries.")
    return float(np.sqrt(_soc_squared_distance(vector)))


def _soc_constraint(vector: cp.Expression, block: slice) -> cp.Constraint:
    # Keep this in DNLP-compatible scalar form.  CVXPY's native SOC
    # Constraint does not itself implement ``is_dnlp`` in CVXPY 1.9.
    return cp.norm(vector[block.start + 1 : block.stop], 2) <= vector[block.start]


def _dimension(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        result = integer_index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a nonnegative integer.") from error
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result


def _dim_value(dims: object, names: Sequence[str], default: Any) -> Any:
    if isinstance(dims, Mapping):
        for name in names:
            if name in dims:
                return dims[name]
        return default
    for name in names:
        if hasattr(dims, name):
            return getattr(dims, name)
    return default


def _has_cones(value: object) -> bool:
    if value is None:
        return False
    if np.isscalar(value):
        return bool(value)
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except TypeError:
        return bool(value)


def _expression_vector(value: cp.Expression | ArrayLike, expected_size: int) -> cp.Expression:
    expression = cp.Expression.cast_to_const(value)
    if not expression.is_real():
        raise ValueError("Cone vectors must be real-valued.")
    if expression.size != expected_size:
        raise ValueError(f"Cone vector has {expression.size} entries; expected {expected_size}.")
    if expression.ndim == 1:
        return expression
    return cp.reshape(expression, (expected_size,), order="F")


def _numeric_vector(value: ArrayLike, expected_size: int) -> NDArray[np.float64]:
    vector = _numeric_vector_unknown_size(value)
    if vector.size != expected_size:
        raise ValueError(f"Cone vector has {vector.size} entries; expected {expected_size}.")
    return vector


def _numeric_vector_unknown_size(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("Cone vectors must be real-valued.")
    try:
        vector = np.asarray(array, dtype=np.float64).reshape(-1, order="F")
    except (TypeError, ValueError) as error:
        raise ValueError("Cone vectors must contain real numeric values.") from error
    return vector


def _nonnegative_squared_distance(vector: NDArray[np.float64]) -> float:
    if not np.all(np.isfinite(vector)):
        return float("inf")
    negative_part = np.minimum(vector, 0.0)
    return float(negative_part @ negative_part)


def _soc_squared_distance(vector: NDArray[np.float64]) -> float:
    if not np.all(np.isfinite(vector)):
        return float("inf")
    head = float(vector[0])
    tail_norm = float(np.linalg.norm(vector[1:]))
    if tail_norm <= head:
        return 0.0
    if tail_norm <= -head:
        return head * head + tail_norm * tail_norm
    distance = (tail_norm - head) / np.sqrt(2.0)
    return float(distance * distance)
