"""Product-cone utilities for BLVPY's supported conic form.

CVXPY orders the supported conic rows as zero, nonnegative, second-order,
exponential, and then three-dimensional power-cone blocks. :class:`ConeLayout`
records that order once and uses it for both symbolic membership constraints
and numerical diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from operator import index as integer_index
from typing import Any, Literal

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

ConeKind = Literal["zero", "nonnegative", "second_order", "exponential", "power_3d"]

_FLOAT_EPSILON = np.finfo(np.float64).eps
_SMALLEST_SUBNORMAL = math.nextafter(0.0, 1.0)
_LOG_SMALLEST_SUBNORMAL = math.log(_SMALLEST_SUBNORMAL)


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
            Euclidean product-cone distance. With finite zero-cone entries,
            nonfinite entries in a constrained block produce positive
            infinity; NaN in a zero-cone block propagates to the result.

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
        for block in self.exponential_slices:
            distance = math.hypot(distance, _exponential_distance(vector[block], dual=False))
        for block, alpha in zip(self.power_3d_slices, self.power_3d, strict=True):
            distance = math.hypot(distance, _power_3d_distance(vector[block], alpha, dual=False))
        return distance

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
        distance = float(np.sqrt(squared_distance))
        for block in self.exponential_slices:
            distance = math.hypot(distance, _exponential_distance(vector[block], dual=True))
        for block, alpha in zip(self.power_3d_slices, self.power_3d, strict=True):
            distance = math.hypot(distance, _power_3d_distance(vector[block], alpha, dual=True))
        return distance

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


def _power_3d_dual_scale(alpha: float) -> float:
    """Return the tail scale in the dual 3D power-cone inequality."""

    complement = 1.0 - alpha
    return math.exp(alpha * math.log(alpha) + complement * math.log(complement))


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


def _binary_normalize(vector: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    """Scale a finite vector by an exact power of two.

    Dividing by an arbitrary largest entry can round away a small, but
    representable, distance from a cone boundary. ``frexp``/``ldexp`` avoid
    that extra rounding for components that remain representable while
    keeping the projection problem at unit scale. Callers retry the raw input
    if the exponent span makes a decisive component underflow.
    """

    magnitude = float(np.max(np.abs(vector)))
    if magnitude == 0.0:
        return vector.copy(), 0
    _, exponent = math.frexp(magnitude)
    # A component can legitimately underflow while the largest component is
    # brought to unit scale.  The distance routines retry the unscaled input
    # whenever that loss could hide a representable residual.
    with np.errstate(under="ignore"):
        return np.ldexp(vector, -exponent), exponent


def _binary_rescale(value: float, exponent: int) -> float:
    """Undo :func:`_binary_normalize` without intermediate overflow."""

    if value == 0.0:
        return 0.0
    try:
        result = math.ldexp(value, exponent)
    except OverflowError:
        return float("inf")
    return result if math.isfinite(result) else float("inf")


def _binary_normalization_is_exact(
    vector: NDArray[np.float64],
    normalized: NDArray[np.float64],
    exponent: int,
) -> bool:
    """Return whether every normalized component round-trips exactly."""

    for original, scaled in zip(vector, normalized, strict=True):
        try:
            restored = math.ldexp(float(scaled), exponent)
        except OverflowError:
            return False
        if restored != float(original):
            return False
    return True


def _decimal_span_precision(*values: float, minimum: int) -> int:
    """Return Decimal precision that covers the binary exponent span."""

    exponents = [math.frexp(abs(value))[1] for value in values if value != 0.0]
    if len(exponents) < 2:
        return minimum
    decimal_span = math.ceil((max(exponents) - min(exponents)) * math.log10(2.0))
    return max(minimum, decimal_span + 80)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide finite scalars, returning a signed infinity on overflow."""

    try:
        return numerator / denominator
    except OverflowError:
        negative = (numerator < 0.0) != (denominator < 0.0)
        return math.copysign(float("inf"), -1.0 if negative else 1.0)


def _log_ratio(numerator: float, denominator: float) -> float:
    """Evaluate ``log(numerator / denominator)`` without ratio overflow.

    The ``log1p`` branch also retains a one-ULP separation when the operands
    are close.  Sterbenz's lemma makes the subtraction exact in that branch.
    """

    difference = numerator - denominator
    if abs(difference) <= 0.5 * denominator:
        return math.log1p(difference / denominator)
    return math.fsum((math.log(numerator), -math.log(denominator)))


def _log_ratio_roundoff(numerator: float, denominator: float, result: float) -> float:
    """Estimate outward roundoff for :func:`_log_ratio`."""

    difference = numerator - denominator
    if abs(difference) <= 0.5 * denominator:
        return math.fsum((8.0 * math.ulp(result), 4.0 * _FLOAT_EPSILON * abs(result)))
    log_numerator = math.log(numerator)
    log_denominator = math.log(denominator)
    return math.fsum(
        (
            4.0 * math.ulp(log_numerator),
            4.0 * math.ulp(log_denominator),
            2.0 * math.ulp(result),
        )
    )


def _exponential_distance(
    vector: NDArray[np.float64],
    *,
    dual: bool,
) -> float:
    """Return Euclidean distance to one primal or dual exponential cone."""

    if not np.all(np.isfinite(vector)):
        return float("inf")
    x_value, y_value, z_value = (float(entry) for entry in vector)
    if not dual and x_value <= 0.0 and y_value <= 0.0:
        return math.hypot(y_value, min(z_value, 0.0))
    if dual and x_value >= 0.0 and y_value >= 0.0:
        return math.hypot(x_value, max(-z_value, 0.0))

    if dual:
        sign_lower_bound = math.hypot(max(x_value, 0.0), min(z_value, 0.0))
    else:
        sign_lower_bound = math.hypot(min(y_value, 0.0), min(z_value, 0.0))

    def raw_distance(normalized_fallback: float | None = None) -> float:
        if dual:
            if _in_exponential_polar(-x_value, -y_value, -z_value):
                return 0.0
            raw_metrics = _exponential_projection_metrics(-vector)
            raw_candidate = None if raw_metrics is None else raw_metrics[2]
        else:
            if _in_exponential_primal(x_value, y_value, z_value):
                return 0.0
            raw_metrics = _exponential_projection_metrics(vector)
            raw_candidate = None if raw_metrics is None else raw_metrics[1]
        if raw_candidate is not None and math.isfinite(raw_candidate) and raw_candidate != 0.0:
            return max(raw_candidate, sign_lower_bound)
        if normalized_fallback is not None and math.isfinite(normalized_fallback) and normalized_fallback != 0.0:
            return max(normalized_fallback, sign_lower_bound)
        if raw_candidate == 0.0:
            # Exact membership was rejected above, so a zero metric here is a
            # certified positive distance below binary64's representable
            # range.  Preserve the nonmembership contract with the least
            # positive result instead of replacing it by a loose face bound.
            return max(_SMALLEST_SUBNORMAL, sign_lower_bound)
        return max(_exponential_feasible_upper_distance(vector, dual=dual), sign_lower_bound)

    normalized, exponent = _binary_normalize(vector)
    if not np.any(normalized):
        return raw_distance()
    normalization_is_exact = _binary_normalization_is_exact(vector, normalized, exponent)
    if dual:
        # Moreau: dist(v, K*) = ||projection_K(-v)||.  The usual coordinate
        # representation K* = {(-y, -x, e*z) in K} is not an isometry.
        metrics = _exponential_projection_metrics(-normalized)
        if metrics is None:
            return raw_distance()
        _, _, normalized_distance = metrics
    else:
        metrics = _exponential_projection_metrics(normalized)
        if metrics is None:
            return raw_distance()
        _, normalized_distance, _ = metrics
    distance = _binary_rescale(normalized_distance, exponent)
    if normalization_is_exact and math.isfinite(distance) and distance != 0.0:
        return max(distance, sign_lower_bound)

    # Scaling a very small component into the subnormal range can retain a
    # nonzero value while still losing low mantissa bits.  Retry at the input
    # scale whenever the transform did not round-trip exactly, as well as for
    # zero or overflowed normalized results.
    return raw_distance(distance)


def _exponential_feasible_upper_distance(
    vector: NDArray[np.float64],
    *,
    dual: bool,
) -> float:
    """Return a feasible-coordinate distance bound after projection failure."""

    x_value, y_value, z_value = (float(entry) for entry in vector)
    if dual:
        candidates = [math.hypot(x_value, min(y_value, 0.0), min(z_value, 0.0))]
        if x_value < 0.0 and z_value > 0.0:
            with localcontext() as context:
                context.prec = 200
                decimal_x = Decimal.from_float(x_value)
                decimal_y = Decimal.from_float(y_value)
                decimal_z = Decimal.from_float(z_value)
                boundary = decimal_x * (Decimal(1) + decimal_z.ln() - (-decimal_x).ln())
                if decimal_y < boundary:
                    candidates.append(float(boundary - decimal_y))
    else:
        candidates = [math.hypot(max(x_value, 0.0), y_value, min(z_value, 0.0))]
        if y_value > 0.0 and z_value > 0.0:
            with localcontext() as context:
                context.prec = 200
                decimal_x = Decimal.from_float(x_value)
                decimal_y = Decimal.from_float(y_value)
                decimal_z = Decimal.from_float(z_value)
                boundary = decimal_y * (decimal_z.ln() - decimal_y.ln())
                if decimal_x > boundary:
                    candidates.append(float(decimal_x - boundary))
    return min(candidates)


def _project_exponential_unit(vector: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Project a finite, normally unit-scaled vector onto the exponential cone."""

    metrics = _exponential_projection_metrics(vector)
    return None if metrics is None else metrics[0]


def _exponential_projection_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Project a finite vector onto the exponential cone.

    The nontrivial smooth-boundary case uses Friberg's monotonically
    increasing scalar equation in ``rho = x / y``.  Its sign is evaluated
    after logarithmic scaling, so neither ``exp(rho)`` nor ``exp(-rho)`` is
    formed during root finding. Inputs are normally power-of-two normalized;
    raw finite inputs are also supported to recover components that disappear
    at unit scale.
    """

    x_value, y_value, z_value = (float(entry) for entry in vector)
    if _in_exponential_primal(x_value, y_value, z_value):
        projected = vector.copy()
        return projected, 0.0, math.hypot(x_value, y_value, z_value)
    if _in_exponential_polar(x_value, y_value, z_value):
        projected = np.zeros(3, dtype=np.float64)
        return projected, math.hypot(x_value, y_value, z_value), 0.0

    # The closest point is on the nonsmooth perspective face throughout this
    # region; this is also the limiting case of the smooth-boundary formula.
    if x_value <= 0.0 and y_value <= 0.0:
        projected_z = max(z_value, 0.0)
        projected = np.array([x_value, 0.0, projected_z], dtype=np.float64)
        distance = math.hypot(y_value, min(z_value, 0.0))
        return projected, distance, math.hypot(x_value, projected_z)

    endpoint_limit = _exponential_endpoint_limit_metrics(vector)
    if endpoint_limit is not None:
        return endpoint_limit

    near_boundary = _exponential_near_boundary_metrics(vector)
    if near_boundary is not None:
        return near_boundary

    lower, upper = _exponential_rho_domain(x_value, y_value)
    if not lower < upper:
        return _exponential_heuristic_projection_metrics(vector)

    def coefficients(rho: float) -> tuple[float, float]:
        rho = float(rho)
        # Factored forms avoid cancellation at finite domain endpoints.
        if x_value == 0.0:
            a_value = y_value
        else:
            a_endpoint = 1.0 - _safe_ratio(y_value, x_value)
            if math.isfinite(a_endpoint):
                if x_value > 0.0:
                    a_value = x_value * (rho - a_endpoint)
                else:
                    a_value = (-x_value) * (a_endpoint - rho)
            else:
                a_value = math.fsum((rho * x_value, y_value, -x_value))

        if y_value == 0.0:
            b_value = x_value
        else:
            b_endpoint = _safe_ratio(x_value, y_value)
            if math.isfinite(b_endpoint):
                if y_value > 0.0:
                    b_value = y_value * (b_endpoint - rho)
                else:
                    b_value = (-y_value) * (rho - b_endpoint)
            else:
                b_value = math.fsum((x_value, -rho * y_value))
        return a_value, b_value

    def boundary_residual(rho: float) -> float:
        rho = float(rho)
        a_value, b_value = coefficients(rho)
        if not a_value > 0.0 or not b_value > 0.0:
            return float("nan")
        log_d = _exponential_log_quadratic(rho)
        signed_logs = [
            (1.0, math.log(a_value) + rho),
            (-1.0, math.log(b_value) - rho),
        ]
        if z_value != 0.0:
            signed_logs.append((-math.copysign(1.0, z_value), log_d + math.log(abs(z_value))))
        largest_log = max(log_value for _, log_value in signed_logs)
        if not math.isfinite(largest_log):
            return float("nan")
        return math.fsum(sign * math.exp(log_value - largest_log) for sign, log_value in signed_logs)

    bracket = _exponential_root_bracket(boundary_residual, lower, upper)
    if bracket is None:
        return _exponential_heuristic_projection_metrics(vector)
    left, right = bracket
    if left == right:
        root = left
    else:
        try:
            root = brentq(
                boundary_residual,
                left,
                right,
                xtol=_SMALLEST_SUBNORMAL,
                rtol=8.0 * _FLOAT_EPSILON,
                maxiter=256,
            )
        except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
            return _exponential_heuristic_projection_metrics(vector)

    def projection_at(rho: float) -> tuple[NDArray[np.float64], float, float] | None:
        a_value, b_value = coefficients(rho)
        if not a_value > 0.0 or not b_value > 0.0:
            return None
        log_d = _exponential_log_quadratic(rho)
        log_y = math.log(a_value) - log_d
        try:
            projected_y = _exp_or_zero(log_y)
            projected_z = _exp_or_zero(math.fsum((rho, log_y)))
            log_b = math.log(b_value)
            x_displacement = _exp_or_zero(log_b - log_d)
            z_displacement = _exp_or_zero(math.fsum((log_b, -rho, -log_d)))
        except OverflowError:
            return None
        projected_x = rho * projected_y
        projected = np.array([projected_x, projected_y, projected_z], dtype=np.float64)
        if not np.all(np.isfinite(projected)):
            return None
        if (x_value <= 0.0 <= projected_x) or (projected_x <= 0.0 <= x_value):
            x_displacement = math.fsum((x_value, -projected_x))
        y_displacement = (1.0 - rho) * x_displacement
        if (y_value <= 0.0 <= projected_y) or (projected_y <= 0.0 <= y_value):
            y_displacement = math.fsum((y_value, -projected_y))
        if (z_value <= 0.0 <= projected_z) or (projected_z <= 0.0 <= z_value):
            z_displacement = math.fsum((projected_z, -z_value))
        distance = math.hypot(x_displacement, y_displacement, z_displacement)
        # The closed-form boundary coordinates independently exponentiate
        # terms that can be enormous and nearly cancel.  Reconstructing them
        # from the KKT displacements retains tiny corrections to a large input
        # and gives a stable norm for the Moreau dual distance.
        stable_projected_x = math.fsum((x_value, -x_displacement))
        stable_projected_y = math.fsum((y_value, -y_displacement))
        stable_projected_z = math.fsum((z_value, z_displacement))
        projection_norm = math.hypot(
            stable_projected_x,
            stable_projected_y,
            stable_projected_z,
        )
        return projected, distance, projection_norm

    # A correctly bracketed root can still land a few ULPs to the wrong side
    # of a near-face KKT condition. Inspect adjacent representable roots and
    # choose the closest candidate whose Moreau decomposition is certified.
    roots = [root]
    lower_neighbor = root
    upper_neighbor = root
    for _ in range(32):
        lower_neighbor = math.nextafter(lower_neighbor, left)
        upper_neighbor = math.nextafter(upper_neighbor, right)
        if lower < lower_neighbor < upper:
            roots.append(lower_neighbor)
        if lower < upper_neighbor < upper:
            roots.append(upper_neighbor)
    candidates = [candidate for rho in roots if (candidate := projection_at(rho)) is not None]
    certified = [candidate for candidate in candidates if _certifies_exponential_projection(vector, candidate[0])]
    if certified:
        return min(certified, key=lambda candidate: candidate[1])
    return _exponential_heuristic_projection_metrics(vector)


def _in_exponential_primal(
    x_value: float,
    y_value: float,
    z_value: float,
    tolerance: float = 0.0,
) -> bool:
    """Test membership in the closed primal EXP cone without exponentiating."""

    x_value, y_value, z_value = float(x_value), float(y_value), float(z_value)
    if y_value < -tolerance or z_value < -tolerance:
        return False
    if y_value <= 0.0:
        return x_value <= tolerance and z_value >= -tolerance
    available_z = z_value + tolerance
    if available_z <= 0.0:
        return False
    if tolerance == 0.0:
        return _exponential_membership_sign(x_value, y_value, z_value, polar=False) >= 0
    boundary_x = y_value * _log_ratio(available_z, y_value)
    return x_value <= boundary_x + tolerance


def _in_exponential_polar(
    x_value: float,
    y_value: float,
    z_value: float,
    tolerance: float = 0.0,
) -> bool:
    """Test membership in the polar of the EXP cone in logarithmic form."""

    x_value, y_value, z_value = float(x_value), float(y_value), float(z_value)
    if x_value < -tolerance or z_value > tolerance:
        return False
    if x_value <= 0.0:
        return y_value <= tolerance and z_value <= tolerance
    available_minus_z = -z_value + tolerance
    if available_minus_z <= 0.0:
        return False
    if tolerance == 0.0:
        return _exponential_membership_sign(x_value, y_value, -z_value, polar=True) >= 0
    boundary_y = x_value * math.fsum((1.0, _log_ratio(available_minus_z, x_value)))
    return y_value <= boundary_y + tolerance


def _exponential_membership_sign(
    first: float,
    second: float,
    positive_third: float,
    *,
    polar: bool,
) -> int:
    """Return an EXP log-slack sign, resolving a rounded boundary."""

    log_ratio = _log_ratio(positive_third, first if polar else second)
    if polar:
        boundary = first * math.fsum((1.0, log_ratio))
        slack = boundary - second
        scale = first
    else:
        boundary = second * log_ratio
        slack = boundary - first
        scale = second
    log_error = _log_ratio_roundoff(
        positive_third,
        first if polar else second,
        log_ratio,
    )
    uncertainty = math.fsum(
        (
            abs(scale) * log_error,
            2.0 * math.ulp(boundary),
            math.ulp(slack),
        )
    )
    if slack > uncertainty:
        return 1
    if slack < -uncertainty:
        return -1

    with localcontext() as context:
        context.prec = 100
        decimal_first = Decimal.from_float(first)
        decimal_second = Decimal.from_float(second)
        decimal_third = Decimal.from_float(positive_third)
        if polar:
            decimal_boundary = decimal_first * (Decimal(1) + decimal_third.ln() - decimal_first.ln())
            decimal_slack = decimal_boundary - decimal_second
        else:
            decimal_boundary = decimal_second * (decimal_third.ln() - decimal_second.ln())
            decimal_slack = decimal_boundary - decimal_first
    return (decimal_slack > 0) - (decimal_slack < 0)


def _exponential_near_boundary_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Return the round-accurate KKT limit next to a smooth EXP face.

    A smooth-boundary displacement can be many orders of magnitude smaller
    than the coordinates themselves.  In that regime no binary64 boundary
    point differs from the input in every required coordinate, so subtracting
    a rounded projection reports a false zero and the scalar root can lose its
    bracket.  Decimal arithmetic retains the exact input floats and evaluates
    the local normal displacement.  The branch is restricted to a tiny
    relative boundary slack, where the omitted curvature term is below the
    precision of the returned binary64 distance.
    """

    x_value, y_value, z_value = (float(entry) for entry in vector)
    relative_limit = 1.0e-12
    with localcontext() as context:
        context.prec = _decimal_span_precision(x_value, y_value, z_value, minimum=200)
        context.Emin = -999_999_999
        context.Emax = 999_999_999
        one = Decimal(1)
        decimal_x = Decimal.from_float(x_value)
        decimal_y = Decimal.from_float(y_value)
        decimal_z = Decimal.from_float(z_value)

        if y_value > 0.0 and z_value > 0.0:
            log_ratio = decimal_z.ln() - decimal_y.ln()
            boundary = decimal_y * log_ratio
            slack = decimal_x - boundary
            scale = max(abs(decimal_x), abs(boundary))
            if slack > 0 and (scale == 0 or slack <= Decimal.from_float(relative_limit) * scale):
                gradient = (one, one - log_ratio, -decimal_y / decimal_z)
                gradient_squared = sum((entry * entry for entry in gradient), Decimal(0))
                multiplier = slack / gradient_squared
                projected_decimal = tuple(
                    entry - multiplier * normal
                    for entry, normal in zip((decimal_x, decimal_y, decimal_z), gradient, strict=True)
                )
                distance = slack / gradient_squared.sqrt()
                projection_norm = sum(
                    (entry * entry for entry in projected_decimal),
                    Decimal(0),
                ).sqrt()
                projected = np.array([float(entry) for entry in projected_decimal], dtype=np.float64)
                if np.all(np.isfinite(projected)):
                    return projected, float(distance), float(projection_norm)

        if x_value > 0.0 and z_value < 0.0:
            positive_third = -decimal_z
            log_ratio = positive_third.ln() - decimal_x.ln()
            boundary = decimal_x * (one + log_ratio)
            slack = decimal_y - boundary
            scale = max(abs(decimal_y), abs(boundary))
            if slack > 0 and (scale == 0 or slack <= Decimal.from_float(relative_limit) * scale):
                gradient = (-log_ratio, one, decimal_x / positive_third)
                gradient_squared = sum((entry * entry for entry in gradient), Decimal(0))
                multiplier = slack / gradient_squared
                projected_decimal = tuple(multiplier * entry for entry in gradient)
                polar_decimal = tuple(
                    entry - projected
                    for entry, projected in zip(
                        (decimal_x, decimal_y, decimal_z),
                        projected_decimal,
                        strict=True,
                    )
                )
                projection_norm = slack / gradient_squared.sqrt()
                distance = sum(
                    (entry * entry for entry in polar_decimal),
                    Decimal(0),
                ).sqrt()
                projected = np.array([float(entry) for entry in projected_decimal], dtype=np.float64)
                if np.all(np.isfinite(projected)):
                    return projected, float(distance), float(projection_norm)
    return None


def _exponential_endpoint_limit_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Certify smooth-face limits beyond binary64's exponential range.

    At either infinite end of the EXP boundary, all non-face corrections can
    round below the smallest subnormal even though the distance itself remains
    representable.  These checks bound the boundary coordinate and each KKT
    correction before returning the corresponding floating-point face limit.
    """

    x_value, y_value, z_value = (float(entry) for entry in vector)
    half_subnormal_log = _LOG_SMALLEST_SUBNORMAL - math.log(2.0)

    if x_value < 0.0 and y_value > 0.0 and z_value <= 0.0:
        rho = _safe_ratio(x_value, y_value)
        if rho == float("-inf"):
            corrections_underflow = True
            log_boundary = float("-inf")
        else:
            log_boundary = math.fsum((math.log(y_value), rho))
            log_tail = float("-inf") if z_value == 0.0 else math.log(-z_value)
            if log_tail == float("-inf"):
                log_distance = log_boundary
            elif log_boundary == float("-inf"):
                log_distance = log_tail
            else:
                larger_log = max(log_tail, log_boundary)
                smaller_log = min(log_tail, log_boundary)
                log_distance = larger_log + math.log1p(math.exp(smaller_log - larger_log))
            # These are logarithms of positive corrections.  At extreme
            # ratios their same-sign sum can lie below binary64's finite
            # range; ordinary addition then saturates to ``-inf``, which is
            # exactly the certified-underflow result needed here.  ``fsum``
            # instead raises on that intermediate overflow.
            log_x_correction = rho + log_distance
            log_y_correction = log_x_correction + math.log1p(-rho)
            corrections_underflow = max(log_x_correction, log_y_correction) < half_subnormal_log
        if corrections_underflow:
            boundary_z = _exp_or_zero(log_boundary)
            projected = np.array([x_value, y_value, boundary_z], dtype=np.float64)
            distance = math.fsum((boundary_z, -z_value))
            return projected, distance, math.hypot(x_value, y_value, boundary_z)

    if x_value > 0.0 and y_value < 0.0:
        with localcontext() as context:
            context.prec = _decimal_span_precision(x_value, y_value, z_value, minimum=200)
            context.Emin = -999_999_999
            context.Emax = 999_999_999
            one = Decimal(1)
            two = Decimal(2)
            decimal_x = Decimal.from_float(x_value)
            decimal_y = Decimal.from_float(y_value)
            decimal_z = Decimal.from_float(z_value)
            endpoint = one - decimal_y / decimal_x

            def residual_at(
                candidate: Decimal,
            ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
                exponential = (-candidate).exp()
                numerator = decimal_z + decimal_x * exponential
                denominator = one + candidate * exponential * exponential
                projected_z = numerator / denominator
                quadratic = candidate * candidate - candidate + one
                residual = decimal_x * (candidate - endpoint) - quadratic * projected_z * exponential
                numerator_derivative = -decimal_x * exponential
                denominator_derivative = exponential * exponential * (one - two * candidate)
                projected_z_derivative = (numerator_derivative * denominator - numerator * denominator_derivative) / (
                    denominator * denominator
                )
                quadratic_derivative = two * candidate - one
                residual_derivative = decimal_x - exponential * (
                    (quadratic_derivative - quadratic) * projected_z + quadratic * projected_z_derivative
                )
                return residual, residual_derivative, exponential, projected_z

            lower = endpoint
            if z_value >= 0.0:
                step = one
                upper = endpoint + step
                for _ in range(16):
                    upper_residual, _, _, _ = residual_at(upper)
                    if upper_residual > 0:
                        break
                    step *= two
                    upper = endpoint + step
                else:
                    upper = endpoint
            else:
                upper = (decimal_x / -decimal_z).ln()

            if z_value > 0.0:
                rho = min(max((decimal_z / decimal_x).ln(), lower), upper)
            else:
                rho = lower
            converged = False
            tolerance = Decimal("1e-80")
            for _ in range(512):
                residual, residual_derivative, exponential, projected_z = residual_at(rho)
                if residual <= 0:
                    lower = rho
                else:
                    upper = rho
                if residual == 0:
                    converged = True
                    break
                if upper - lower <= tolerance * max(one, abs(rho)):
                    rho = (lower + upper) / two
                    converged = True
                    break
                updated = rho - residual / residual_derivative if residual_derivative > 0 else (lower + upper) / two
                if updated == rho:
                    converged = True
                    break
                if not lower < updated < upper:
                    updated = endpoint + (rho * rho - rho + one) * projected_z * exponential / decimal_x
                    if updated == rho:
                        converged = True
                        break
                    if not lower < updated < upper:
                        updated = (lower + upper) / two
                rho = updated

            _, _, exponential, projected_z = residual_at(rho)
            if converged and projected_z > 0:
                projected_y = projected_z * exponential
                projected_x = rho * projected_y
                projected_decimal = (projected_x, projected_y, projected_z)
                displacement_decimal = (
                    decimal_x - projected_x,
                    decimal_y - projected_y,
                    decimal_z - projected_z,
                )
                distance = sum(
                    (entry * entry for entry in displacement_decimal),
                    Decimal(0),
                ).sqrt()
                projection_norm = sum(
                    (entry * entry for entry in projected_decimal),
                    Decimal(0),
                ).sqrt()
                projected = np.array([float(entry) for entry in projected_decimal], dtype=np.float64)
                return projected, float(distance), float(projection_norm)

    if x_value > 0.0 and y_value < 0.0 and z_value >= 0.0:
        ratio = _safe_ratio(-y_value, x_value)
        rho = math.fsum((1.0, ratio)) if math.isfinite(ratio) else float("inf")
        if rho == float("inf"):
            corrections_underflow = True
        else:
            log_rho = math.log(rho)
            log_projected_y = float("-inf") if z_value == 0.0 else math.fsum((math.log(z_value), -rho))
            log_projected_x = math.fsum((log_projected_y, log_rho))
            log_z_correction = math.fsum((math.log(x_value), -rho))
            corrections_underflow = (
                max(
                    log_projected_x,
                    log_projected_y,
                    log_z_correction,
                )
                < half_subnormal_log
            )
        if corrections_underflow:
            projected = np.array([0.0, 0.0, z_value], dtype=np.float64)
            return projected, math.hypot(x_value, y_value), z_value
    return None


def _exponential_heuristic_projection(vector: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Return a certified limiting projection when ``rho`` is unresolvable."""

    metrics = _exponential_heuristic_projection_metrics(vector)
    return None if metrics is None else metrics[0]


def _exponential_heuristic_projection_metrics(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float] | None:
    """Return certified projection metrics when ``rho`` is unresolvable."""

    x_value, y_value, z_value = (float(entry) for entry in vector)
    face = np.array([min(x_value, 0.0), 0.0, max(z_value, 0.0)], dtype=np.float64)
    candidates = [
        (
            face,
            math.hypot(max(x_value, 0.0), y_value, min(z_value, 0.0)),
            math.hypot(float(face[0]), float(face[2])),
        )
    ]
    if y_value > 0.0:
        log_boundary_z = math.fsum((math.log(y_value), _safe_ratio(x_value, y_value)))
        if log_boundary_z <= math.log(np.finfo(np.float64).max):
            boundary_z = _exp_or_zero(log_boundary_z)
            projected_z = max(z_value, boundary_z)
            candidate = np.array([x_value, y_value, projected_z], dtype=np.float64)
            candidates.append(
                (
                    candidate,
                    projected_z - z_value,
                    math.hypot(x_value, y_value, projected_z),
                )
            )

    certified = [
        candidate
        for candidate in candidates
        if candidate[1] > 0.0 and _certifies_exponential_projection(vector, candidate[0])
    ]
    if not certified:
        return None
    return min(certified, key=lambda candidate: candidate[1])


def _certifies_exponential_projection(
    vector: NDArray[np.float64],
    projected: NDArray[np.float64],
) -> bool:
    """Check approximate primal feasibility and Moreau optimality."""

    if not np.any(projected):
        return _in_exponential_polar(*vector)

    certification_tolerance = 2e-10
    polar = vector - projected
    if not _in_exponential_primal(*projected, certification_tolerance):
        return False
    if not _in_exponential_polar(*polar, certification_tolerance):
        return False
    projected_scale = max(abs(float(entry)) for entry in projected)
    polar_scale = max(abs(float(entry)) for entry in polar)
    if projected_scale == 0.0 or polar_scale == 0.0:
        return True

    # Compare the pairing in normalized coordinates.  Computing either the
    # raw dot product or the product of norms can overflow for finite vectors,
    # turning the old ``inf <= inf`` check into a false certificate.
    projected_unit = tuple(float(entry) / projected_scale for entry in projected)
    polar_unit = tuple(float(entry) / polar_scale for entry in polar)
    pairing = math.fsum(
        projected_entry * polar_entry for projected_entry, polar_entry in zip(projected_unit, polar_unit, strict=True)
    )
    if pairing == 0.0:
        return True
    norm_product = math.hypot(*projected_unit) * math.hypot(*polar_unit)
    magnitude_log = math.log(projected_scale) + math.log(polar_scale)
    pairing_log = math.log(abs(pairing)) + magnitude_log
    scale_log = max(0.0, math.log(norm_product) + magnitude_log)
    return pairing_log <= math.log(certification_tolerance) + scale_log


def _exponential_rho_domain(x_value: float, y_value: float) -> tuple[float, float]:
    """Return the open interval where Friberg's two linear factors are positive."""

    lower = float("-inf")
    upper = float("inf")
    if x_value > 0.0:
        lower = max(lower, 1.0 - _safe_ratio(y_value, x_value))
    elif x_value < 0.0:
        upper = min(upper, 1.0 - _safe_ratio(y_value, x_value))
    elif y_value <= 0.0:
        return 0.0, 0.0

    if y_value > 0.0:
        upper = min(upper, _safe_ratio(x_value, y_value))
    elif y_value < 0.0:
        lower = max(lower, _safe_ratio(x_value, y_value))
    elif x_value <= 0.0:
        return 0.0, 0.0
    return lower, upper


def _exponential_root_bracket(
    function: Any,
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    """Bracket the increasing EXP projection equation on an open interval."""

    if lower < 0.0 < upper:
        center = 0.0
    elif math.isfinite(lower) and lower >= 0.0:
        width = upper - lower
        offset = min(1.0, 0.5 * width) if math.isfinite(width) else 1.0
        center = lower + offset
        if center <= lower or center >= upper:
            center = math.nextafter(lower, upper)
    elif math.isfinite(upper) and upper <= 0.0:
        width = upper - lower
        offset = min(1.0, 0.5 * width) if math.isfinite(width) else 1.0
        center = upper - offset
        if center <= lower or center >= upper:
            center = math.nextafter(upper, lower)
    else:
        center = 0.0

    center_value = function(center)
    if not math.isfinite(center_value):
        # Move toward the middle of the open interval until both linear
        # factors are representably positive.
        for _ in range(64):
            if math.isfinite(lower) and math.isfinite(upper):
                center = math.nextafter(center, upper)
            elif math.isfinite(lower):
                center = center + max(1.0, abs(center)) * _FLOAT_EPSILON
            else:
                center = center - max(1.0, abs(center)) * _FLOAT_EPSILON
            center_value = function(center)
            if math.isfinite(center_value):
                break
        else:
            return None
    if center_value == 0.0:
        return center, center

    if center_value > 0.0:
        right = center
        left = center
        step = max(1.0, 0.5 * abs(center))
        for _ in range(1024):
            candidate = left - step
            reached_endpoint = math.isfinite(lower) and candidate <= lower
            if reached_endpoint:
                candidate = left + 0.5 * (lower - left)
                if candidate <= lower or candidate >= left:
                    candidate = math.nextafter(lower, upper)
            if not math.isfinite(candidate):
                return None
            value = function(candidate)
            if math.isfinite(value) and value < 0.0:
                return candidate, right
            if reached_endpoint and candidate == math.nextafter(lower, upper):
                return None
            left = candidate
            step *= 2.0
            if not math.isfinite(step):
                return None
        return None

    left = center
    right = center
    step = max(1.0, 0.5 * abs(center))
    for _ in range(1024):
        candidate = right + step
        reached_endpoint = math.isfinite(upper) and candidate >= upper
        if reached_endpoint:
            candidate = right + 0.5 * (upper - right)
            if candidate <= right or candidate >= upper:
                candidate = math.nextafter(upper, lower)
        if not math.isfinite(candidate):
            return None
        value = function(candidate)
        if math.isfinite(value) and value > 0.0:
            return left, candidate
        if reached_endpoint and candidate == math.nextafter(upper, lower):
            return None
        right = candidate
        step *= 2.0
        if not math.isfinite(step):
            return None
    return None


def _exponential_log_quadratic(rho: float) -> float:
    """Evaluate ``log(rho**2 - rho + 1)`` without overflow."""

    absolute = abs(rho)
    if absolute < 1e150:
        return math.log(rho * (rho - 1.0) + 1.0)
    inverse = 1.0 / rho
    return 2.0 * math.log(absolute) + math.log1p(-inverse + inverse * inverse)


def _power_3d_distance(
    vector: NDArray[np.float64],
    alpha: float,
    *,
    dual: bool,
) -> float:
    """Return Euclidean distance to one primal or dual power cone."""

    if not np.all(np.isfinite(vector)):
        return float("inf")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return 0.0
    normalized = vector / scale
    if dual:
        # Moreau: dist(v, K*) = ||projection_K(-v)||. Scaling the first two
        # coordinates describes K* symbolically but is not an isometry.
        projected = _project_power_3d_unit(-normalized, alpha)
        if projected is None:
            return float("inf")
        normalized_distance = math.hypot(float(projected[0]), float(projected[1]), float(projected[2]))
    else:
        projected = _project_power_3d_unit(normalized, alpha)
        if projected is None:
            return float("inf")
        difference = normalized - projected
        normalized_distance = math.hypot(float(difference[0]), float(difference[1]), float(difference[2]))
    if normalized_distance == 0.0:
        return 0.0
    distance = scale * normalized_distance
    return distance if math.isfinite(distance) else float("inf")


def _project_power_3d_unit(
    vector: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64] | None:
    """Project a finite, unit-scaled vector onto a 3D power cone.

    The nontrivial boundary case is Hien's scalar equation. It is evaluated
    in logarithmic coordinates and solved in a logit parameter with a
    bracketed Brent iteration. The logit parameter continues to distinguish
    roots whose tail coordinate rounds to either zero or its input value.
    """

    x_head, y_head, z_head = (float(entry) for entry in vector)
    tail = abs(z_head)
    complement = 1.0 - alpha
    comparison_tolerance = 16.0 * _FLOAT_EPSILON

    if x_head >= 0.0 and y_head >= 0.0:
        if tail == 0.0:
            return vector.copy()
        log_product = _power_3d_log_product(x_head, y_head, alpha)
        if math.log(tail) <= log_product + comparison_tolerance:
            return vector.copy()

    if x_head <= 0.0 and y_head <= 0.0:
        if tail == 0.0:
            return np.zeros(3, dtype=np.float64)
        log_product = _power_3d_log_product(-x_head, -y_head, alpha)
        log_dual_tail = math.log(tail) + math.log(_power_3d_dual_scale(alpha))
        if log_dual_tail <= log_product + comparison_tolerance:
            return np.zeros(3, dtype=np.float64)

    if tail == 0.0:
        return np.array([max(x_head, 0.0), max(y_head, 0.0), 0.0], dtype=np.float64)

    log_tail = math.log(tail)

    def boundary_residual(logit: float) -> float:
        log_fraction = _log_sigmoid(logit)
        log_complement_fraction = _log_sigmoid(-logit)
        log_projected_tail = log_tail + log_fraction
        log_tail_reduction = log_tail + log_complement_fraction
        log_x = _power_3d_projected_head_log(
            x_head,
            alpha,
            log_projected_tail,
            log_tail_reduction,
        )
        log_y = _power_3d_projected_head_log(
            y_head,
            complement,
            log_projected_tail,
            log_tail_reduction,
        )
        return alpha * log_x + complement * log_y - log_projected_tail

    left = -1.0
    left_value = boundary_residual(left)
    for _ in range(1024):
        if np.isfinite(left_value) and left_value > 0.0:
            break
        left *= 2.0
        if not np.isfinite(left):
            return None
        left_value = boundary_residual(left)
    else:
        return None

    right = 1.0
    right_value = boundary_residual(right)
    for _ in range(1024):
        if np.isfinite(right_value) and right_value < 0.0:
            break
        right *= 2.0
        if not np.isfinite(right):
            return None
        right_value = boundary_residual(right)
    else:
        return None

    try:
        root = brentq(
            boundary_residual,
            left,
            right,
            xtol=1e-12,
            rtol=8.0 * _FLOAT_EPSILON,
            maxiter=256,
        )
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        return None

    log_fraction = _log_sigmoid(root)
    log_complement_fraction = _log_sigmoid(-root)
    log_projected_tail = log_tail + log_fraction
    log_tail_reduction = log_tail + log_complement_fraction
    log_x = _power_3d_projected_head_log(
        x_head,
        alpha,
        log_projected_tail,
        log_tail_reduction,
    )
    log_y = _power_3d_projected_head_log(
        y_head,
        complement,
        log_projected_tail,
        log_tail_reduction,
    )
    projected = np.array(
        [
            _exp_or_zero(log_x),
            _exp_or_zero(log_y),
            math.copysign(tail * math.exp(log_fraction), z_head),
        ],
        dtype=np.float64,
    )
    return projected if np.all(np.isfinite(projected)) else None


def _power_3d_log_product(x: float, y: float, alpha: float) -> float:
    if x <= 0.0 or y <= 0.0:
        return float("-inf")
    return alpha * math.log(x) + (1.0 - alpha) * math.log(y)


def _power_3d_projected_head_log(
    head: float,
    weight: float,
    log_projected_tail: float,
    log_tail_reduction: float,
) -> float:
    """Evaluate log((head + sqrt(head**2 + 4*weight*q))/2)."""

    log_q = log_projected_tail + log_tail_reduction
    log_sqrt_term = 0.5 * (math.log(4.0 * weight) + log_q)
    sqrt_term = 0.0 if log_sqrt_term < _LOG_SMALLEST_SUBNORMAL else math.exp(log_sqrt_term)
    discriminant_root = math.hypot(head, sqrt_term)
    if head > 0.0:
        return math.log(0.5 * (head + discriminant_root))
    if head == 0.0:
        return 0.5 * (math.log(weight) + log_q)
    # Rationalize the numerator to avoid cancellation for a negative head.
    return math.log(2.0 * weight) + log_q - math.log(discriminant_root - head)


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _exp_or_zero(value: float) -> float:
    return 0.0 if value < _LOG_SMALLEST_SUBNORMAL else math.exp(value)
