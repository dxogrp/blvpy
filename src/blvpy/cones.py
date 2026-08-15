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
    """One contiguous cone block in a canonical constraint vector."""

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
        """Number of scalar rows in the block."""

        return self.stop - self.start

    @property
    def slice(self) -> slice:
        """Slice selecting this block from a canonical vector."""

        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class ConeLayout:
    """Immutable layout of a zero/nonnegative/SOC product cone.

    Parameters follow the canonical CVXPY row order.  A second-order cone
    size includes its scalar head, so every entry in ``second_order`` must be
    at least two.
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
        """Build a layout from CVXPY ``ConeDims`` or an equivalent mapping.

        Unsupported nonempty cone dimensions are rejected here, rather than
        being omitted from the canonical row count.
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
        """CVXPY-compatible alias for :attr:`nonnegative`."""

        return self.nonnegative

    @property
    def soc(self) -> tuple[int, ...]:
        """CVXPY-compatible alias for :attr:`second_order`."""

        return self.second_order

    @property
    def size(self) -> int:
        """Total number of scalar cone rows."""

        return self.zero + self.nonnegative + sum(self.second_order)

    @property
    def zero_slice(self) -> slice:
        """Slice of the zero-cone rows, possibly empty."""

        return slice(0, self.zero)

    @property
    def nonnegative_slice(self) -> slice:
        """Slice of the nonnegative-cone rows, possibly empty."""

        return slice(self.zero, self.zero + self.nonnegative)

    @property
    def nonneg_slice(self) -> slice:
        """CVXPY-compatible alias for :attr:`nonnegative_slice`."""

        return self.nonnegative_slice

    @property
    def second_order_slices(self) -> tuple[slice, ...]:
        """Slices of the ordered second-order cone blocks."""

        start = self.zero + self.nonnegative
        slices: list[slice] = []
        for size in self.second_order:
            slices.append(slice(start, start + size))
            start += size
        return tuple(slices)

    @property
    def soc_slices(self) -> tuple[slice, ...]:
        """CVXPY-compatible alias for :attr:`second_order_slices`."""

        return self.second_order_slices

    @property
    def blocks(self) -> tuple[ConeBlock, ...]:
        """All nonempty blocks in canonical row order."""

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
        """Return constraints imposing membership in the primal cone."""

        vector = _expression_vector(value, self.size)
        constraints: list[cp.Constraint] = []
        if self.zero:
            constraints.append(vector[self.zero_slice] == 0)
        if self.nonnegative:
            constraints.append(vector[self.nonnegative_slice] >= 0)
        constraints.extend(_soc_constraint(vector, block) for block in self.second_order_slices)
        return tuple(constraints)

    def dual_constraints(self, value: cp.Expression | ArrayLike) -> tuple[cp.Constraint, ...]:
        """Return constraints imposing membership in the dual product cone.

        The dual of the zero cone is the whole space, while the nonnegative
        and second-order cones are self-dual.
        """

        vector = _expression_vector(value, self.size)
        constraints: list[cp.Constraint] = []
        if self.nonnegative:
            constraints.append(vector[self.nonnegative_slice] >= 0)
        constraints.extend(_soc_constraint(vector, block) for block in self.second_order_slices)
        return tuple(constraints)

    def primal_distance(self, value: ArrayLike) -> float:
        """Euclidean distance from ``value`` to the primal product cone."""

        vector = _numeric_vector(value, self.size)
        squared_distance = float(np.dot(vector[self.zero_slice], vector[self.zero_slice]))
        squared_distance += _nonnegative_squared_distance(vector[self.nonnegative_slice])
        squared_distance += sum(_soc_squared_distance(vector[block]) for block in self.second_order_slices)
        return float(np.sqrt(squared_distance))

    def dual_distance(self, value: ArrayLike) -> float:
        """Euclidean distance from ``value`` to the dual product cone."""

        vector = _numeric_vector(value, self.size)
        squared_distance = _nonnegative_squared_distance(vector[self.nonnegative_slice])
        squared_distance += sum(_soc_squared_distance(vector[block]) for block in self.second_order_slices)
        return float(np.sqrt(squared_distance))

    def complementarity(self, primal: ArrayLike, dual: ArrayLike) -> float:
        """Return the canonical Euclidean pairing ``primal @ dual``."""

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
