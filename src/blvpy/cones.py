"""Product-cone utilities for BLVPY's supported conic form.

CVXPY orders the supported conic rows as zero, nonnegative, second-order,
exponential, and then three-dimensional power-cone blocks. :class:`ConeLayout`
records that order once and uses it for both symbolic membership constraints
and numerical diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Any, Literal

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cones.numeric import _saturated_hypot
from ._cones.power import _power_3d_dual_scale
from ._cones.projection import _nonlinear_cone_distance

ConeKind = Literal["zero", "nonnegative", "second_order", "exponential", "power_3d"]


@dataclass(frozen=True, slots=True)
class ConeBlock:
    """One contiguous block in a canonical product-cone vector.

    Parameters
    ----------
    kind : {"zero", "nonnegative", "second_order", "exponential", "power_3d"}
        Cone represented by the block.
    start : int
        Inclusive zero-based row offset.
    stop : int
        Exclusive row offset; it must be greater than ``start``.
    index : int, default=0
        Zero-based index among blocks of the same kind. It distinguishes
        multiple cones of one kind.

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
        if self.kind not in {"zero", "nonnegative", "second_order", "exponential", "power_3d"}:
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
    """Ordered layout of a supported product cone.

    Parameters
    ----------
    zero : int, default=0
        Number of scalar rows in the zero-cone block.
    nonnegative : int, default=0
        Number of scalar rows in the nonnegative-cone block.
    second_order : tuple of int, optional
        Dimensions of the second-order cone blocks. Each dimension includes
        the scalar head and must be at least two.
    power_3d : tuple of float, optional
        Exponent ``alpha`` for each three-dimensional power-cone block. Every
        exponent must be finite and lie strictly between zero and one.
    exponential : int, default=0
        Number of three-dimensional exponential-cone blocks. This field is
        declared after ``power_3d`` to preserve existing positional calls,
        while its rows precede power-cone rows in canonical order.

    Raises
    ------
    ValueError
        If a dimension or power-cone exponent is invalid.

    Notes
    -----
    Rows follow CVXPY's canonical order: zero, nonnegative, each SOC in
    sequence, each exponential cone, and each 3D power cone in sequence. The
    associated dual cone is unrestricted on zero-cone rows and self-dual on
    nonnegative and SOC rows. Exponential and power cones use their respective
    nonsymmetric duals.
    """

    zero: int = 0
    nonnegative: int = 0
    second_order: tuple[int, ...] = ()
    power_3d: tuple[float, ...] = ()
    exponential: int = 0

    def __post_init__(self) -> None:
        zero = _dimension(self.zero, "zero")
        nonnegative = _dimension(self.nonnegative, "nonnegative")
        exponential = _dimension(self.exponential, "exponential")
        try:
            second_order = tuple(
                _dimension(size, f"second_order[{position}]") for position, size in enumerate(self.second_order)
            )
        except TypeError as error:
            raise ValueError("second_order must be a sequence of cone sizes.") from error
        if any(size < 2 for size in second_order):
            raise ValueError("Every second-order cone must have dimension at least 2.")
        try:
            power_3d = tuple(
                _power_3d_exponent(alpha, f"power_3d[{position}]") for position, alpha in enumerate(self.power_3d)
            )
        except TypeError as error:
            raise ValueError("power_3d must be a sequence of cone exponents.") from error
        object.__setattr__(self, "zero", zero)
        object.__setattr__(self, "nonnegative", nonnegative)
        object.__setattr__(self, "second_order", second_order)
        object.__setattr__(self, "power_3d", power_3d)
        object.__setattr__(self, "exponential", exponential)

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
            Validated supported product-cone row layout.

        Raises
        ------
        ValueError
            If ``dims`` is ``None``, contains invalid dimensions, or declares
            nonempty PSD or N-dimensional power cones.
        """

        if dims is None:
            raise ValueError("Cone dimensions cannot be None.")

        zero = _dim_value(dims, ("zero", "f"), 0)
        nonnegative = _dim_value(dims, ("nonnegative", "nonneg", "l"), 0)
        second_order = _dim_value(dims, ("second_order", "soc", "q"), ())
        power_3d = _dim_value(dims, ("power_3d", "p3d", "p3"), ())
        exponential = _dim_value(dims, ("exponential", "exp", "ep"), 0)

        unsupported: list[str] = []
        for label, names in (
            ("positive-semidefinite", ("psd", "s")),
            ("N-dimensional power", ("pnd",)),
        ):
            value = _dim_value(dims, names, 0)
            if _has_cones(value):
                unsupported.append(label)
        if unsupported:
            rendered = ", ".join(unsupported)
            raise ValueError(f"Unsupported cone dimensions: {rendered}.")

        return cls(
            zero=zero,
            nonnegative=nonnegative,
            second_order=tuple(second_order or ()),
            power_3d=power_3d if power_3d is not None else (),
            exponential=exponential,
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
    def p3d(self) -> tuple[float, ...]:
        """tuple of float: CVXPY-compatible alias for ``power_3d``."""

        return self.power_3d

    @property
    def exp(self) -> int:
        """int: CVXPY-compatible alias for ``exponential``."""

        return self.exponential

    @property
    def size(self) -> int:
        """int: Total number of scalar product-cone rows."""

        return self.zero + self.nonnegative + sum(self.second_order) + 3 * self.exponential + 3 * len(self.power_3d)

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
    def power_3d_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Ordered three-dimensional power-cone row slices."""

        start = self.zero + self.nonnegative + sum(self.second_order) + 3 * self.exponential
        return tuple(slice(start + 3 * position, start + 3 * (position + 1)) for position in range(len(self.power_3d)))

    @property
    def p3d_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Alias for ``power_3d_slices``."""

        return self.power_3d_slices

    @property
    def exponential_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Ordered three-dimensional exponential-cone rows."""

        start = self.zero + self.nonnegative + sum(self.second_order)
        return tuple(slice(start + 3 * position, start + 3 * (position + 1)) for position in range(self.exponential))

    @property
    def exp_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Alias for ``exponential_slices``."""

        return self.exponential_slices

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
        blocks.extend(
            ConeBlock("exponential", block.start, block.stop, position)
            for position, block in enumerate(self.exponential_slices)
        )
        blocks.extend(
            ConeBlock("power_3d", block.start, block.stop, position)
            for position, block in enumerate(self.power_3d_slices)
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
            Zero equalities, nonnegative inequalities, scalar-form SOC
            inequalities, and exact exponential- and 3D power-cone constraints
            in canonical block order.

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
        for block in self.exponential_slices:
            constraints.extend(_exponential_constraints(vector, block, dual=False))
        for block, alpha in zip(self.power_3d_slices, self.power_3d, strict=True):
            constraints.extend(_power_3d_constraints(vector, block, alpha, dual=False))
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
            Nonnegative, SOC, dual exponential-cone, and dual 3D power-cone
            membership constraints. Zero-cone dual rows are unrestricted and
            therefore add no constraints.

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
        for block in self.exponential_slices:
            constraints.extend(_exponential_constraints(vector, block, dual=True))
        for block, alpha in zip(self.power_3d_slices, self.power_3d, strict=True):
            constraints.extend(_power_3d_constraints(vector, block, alpha, dual=True))
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
            Numerical Euclidean product-cone distance. Zero, nonnegative, and
            second-order contributions are analytic. Exponential and 3D
            power-cone contributions are auxiliary-solver estimates; a value
            below normalized solver resolution may be reported as zero, and
            an unusable projection produces a conservative finite upper
            bound. With finite zero-cone entries, nonfinite entries in a
            constrained block produce positive infinity; NaN in a zero-cone
            block propagates to the result.

        Raises
        ------
        ValueError
            If ``value`` is complex, nonnumeric, or has the wrong size.
        """

        vector = _numeric_vector(value, self.size)
        squared_distance = float(np.dot(vector[self.zero_slice], vector[self.zero_slice]))
        squared_distance += _nonnegative_squared_distance(vector[self.nonnegative_slice])
        squared_distance += sum(_soc_squared_distance(vector[block]) for block in self.second_order_slices)
        distance = float(np.sqrt(squared_distance))
        nonlinear_distance = _nonlinear_cone_distance(
            tuple(vector[block] for block in self.exponential_slices),
            tuple((vector[block], alpha) for block, alpha in zip(self.power_3d_slices, self.power_3d, strict=True)),
            dual=False,
        )
        return _saturated_hypot(distance, nonlinear_distance)

    def dual_distance(self, value: ArrayLike) -> float:
        """Compute distance to the dual product cone.

        Parameters
        ----------
        value : array-like
            Real numeric vector with :attr:`size` entries.

        Returns
        -------
        float
            Numerical Euclidean product-cone distance, with zero-cone dual
            rows unrestricted. Exponential and 3D power-cone contributions
            are auxiliary-solver estimates; a value below normalized solver
            resolution may be reported as zero, and an unusable projection
            produces a conservative finite upper bound. Nonfinite constrained
            entries produce positive infinity.

        Raises
        ------
        ValueError
            If ``value`` is complex, nonnumeric, or has the wrong size.
        """

        vector = _numeric_vector(value, self.size)
        squared_distance = _nonnegative_squared_distance(vector[self.nonnegative_slice])
        squared_distance += sum(_soc_squared_distance(vector[block]) for block in self.second_order_slices)
        distance = float(np.sqrt(squared_distance))
        nonlinear_distance = _nonlinear_cone_distance(
            tuple(vector[block] for block in self.exponential_slices),
            tuple((vector[block], alpha) for block, alpha in zip(self.power_3d_slices, self.power_3d, strict=True)),
            dual=True,
        )
        return _saturated_hypot(distance, nonlinear_distance)

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
    # Keep this in DNLP-compliant scalar form.  CVXPY's native SOC
    # Constraint does not itself implement ``is_dnlp`` in CVXPY 1.9.
    return cp.norm(vector[block.start + 1 : block.stop], 2) <= vector[block.start]


def _exponential_constraints(
    vector: cp.Expression,
    block: slice,
    *,
    dual: bool,
) -> tuple[cp.Constraint, ...]:
    """Construct exact scalar membership constraints for one EXP block."""

    x = vector[block.start]
    y = vector[block.start + 1]
    z = vector[block.start + 2]
    # Native ExpCone constraints do not implement is_dnlp in CVXPY 1.9.
    # Relative entropy gives exact closed-cone descriptions that are both DCP
    # and DNLP, including the y=0 (or, dually, x=0) closure faces.
    if dual:
        return x <= 0, z >= 0, cp.rel_entr(-x, z) <= y - x
    return y >= 0, z >= 0, cp.rel_entr(y, z) <= -x


def _power_3d_constraints(
    vector: cp.Expression,
    block: slice,
    alpha: float,
    *,
    dual: bool,
) -> tuple[cp.Constraint, ...]:
    x = vector[block.start]
    y = vector[block.start + 1]
    z = vector[block.start + 2]
    geometric_mean = cp.geo_mean(
        cp.hstack([x, y]),
        p=[alpha, 1.0 - alpha],
        approx=False,
    )
    tail_scale = _power_3d_dual_scale(alpha) if dual else 1.0
    # Native PowCone3D constraints do not implement is_dnlp in CVXPY 1.9.
    # This exact scalar form is both DCP (for fixed-lower solves) and DNLP.
    return x >= 0, y >= 0, tail_scale * cp.abs(z) <= geometric_mean


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


def _power_3d_exponent(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real number strictly between zero and one.")
    try:
        array = np.asarray(value, dtype=np.float64)
        if array.shape:
            raise ValueError
        result = float(array)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number strictly between zero and one.") from error
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be a finite real number strictly between zero and one.")
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
