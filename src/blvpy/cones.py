"""Product-cone utilities for BLVPY's supported conic form.

CVXPY orders the supported conic rows as zero, nonnegative, second-order,
and then three-dimensional power-cone blocks. :class:`ConeLayout` records
that order once and uses it for both symbolic membership constraints and
numerical diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Any, Literal

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

ConeKind = Literal["zero", "nonnegative", "second_order", "power_3d"]

_FLOAT_EPSILON = np.finfo(np.float64).eps
_LOG_SMALLEST_SUBNORMAL = math.log(float(np.nextafter(0.0, 1.0)))


@dataclass(frozen=True, slots=True)
class ConeBlock:
    """One contiguous block in a canonical product-cone vector.

    Parameters
    ----------
    kind : {"zero", "nonnegative", "second_order", "power_3d"}
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
        if self.kind not in {"zero", "nonnegative", "second_order", "power_3d"}:
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

    Raises
    ------
    ValueError
        If a dimension or power-cone exponent is invalid.

    Notes
    -----
    Rows follow CVXPY's canonical order: zero, nonnegative, each SOC in
    sequence, and each 3D power cone in sequence. The associated dual cone is
    unrestricted on zero-cone rows and self-dual on nonnegative and SOC rows.
    A power cone with exponent ``alpha`` has the scaled power cone as its dual.
    """

    zero: int = 0
    nonnegative: int = 0
    second_order: tuple[int, ...] = ()
    power_3d: tuple[float, ...] = ()

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
            nonempty PSD, exponential, or N-dimensional power cones.
        """

        if dims is None:
            raise ValueError("Cone dimensions cannot be None.")

        zero = _dim_value(dims, ("zero", "f"), 0)
        nonnegative = _dim_value(dims, ("nonnegative", "nonneg", "l"), 0)
        second_order = _dim_value(dims, ("second_order", "soc", "q"), ())
        power_3d = _dim_value(dims, ("power_3d", "p3d", "p3"), ())

        unsupported: list[str] = []
        for label, names in (
            ("exponential", ("exp", "ep")),
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
    def size(self) -> int:
        """int: Total number of scalar product-cone rows."""

        return self.zero + self.nonnegative + sum(self.second_order) + 3 * len(self.power_3d)

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

        start = self.zero + self.nonnegative + sum(self.second_order)
        return tuple(slice(start + 3 * position, start + 3 * (position + 1)) for position in range(len(self.power_3d)))

    @property
    def p3d_slices(self) -> tuple[slice, ...]:
        """tuple of slice: Alias for ``power_3d_slices``."""

        return self.power_3d_slices

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
            inequalities, and exact 3D power-cone constraints in canonical
            block order.

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
            Nonnegative, SOC, and dual 3D power-cone membership constraints.
            Zero-cone dual rows are unrestricted and therefore add no
            constraints.

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
