"""Public exception hierarchy for BLVPY."""

from __future__ import annotations


class BilevelError(Exception):
    """Base class for BLVPY-specific errors.

    Catch this class to handle any modeled validation, canonicalization,
    initialization, or solve failure emitted through BLVPY's public API.
    Ordinary CVXPY and Python argument errors may still propagate when they do
    not belong to a BLVPY-specific boundary.
    """


class ValidationError(BilevelError, ValueError):
    """Raised when a model fails disciplined bilevel validation.

    This includes invalid upper/lower structure and lower DCP or DPP failures.
    More specific unsupported-model and parameter-linking failures derive from
    this class.
    """


class ParameterMappingError(ValidationError):
    """Raised when an upper variable cannot parameterize the lower problem.

    Examples include duplicate or unused ``LowerProblem.parameters``, missing
    fixed parameter values, incompatible shapes, and invalid canonical
    parameter packing.
    """


class UnsupportedModelError(ValidationError):
    """Raised when a model uses a feature outside BLVPY's supported subset.

    Unsupported variable domains, source atoms, and lifted expressions that
    are not DNLP-compliant are reported through this exception.
    """


class UnsupportedConeError(UnsupportedModelError):
    """Raised when lower canonicalization produces an unsupported cone.

    BLVPY 0.1 supports zero, nonnegative, and second-order cones. PSD,
    exponential, and power-cone blocks trigger this exception before nonlinear
    solving.
    """


class ApproximateCanonicalizationError(UnsupportedModelError):
    """Raised when a source expression has only an approximate cone graph.

    The SOCP reformulation requires an audited pointwise-exact
    canonicalization; approximation-based atoms and constraints are rejected.
    """


class CanonicalizationError(BilevelError, RuntimeError):
    """Raised when CVXPY cannot expose BLVPY's expected canonical program.

    This signals a failure or unexpected reduction structure after source-level
    validation, rather than an unsupported cone reported by
    :class:`blvpy.UnsupportedConeError`.
    """


class InitializationError(BilevelError, RuntimeError):
    """Raised when no acceptable initial continuation point can be built.

    The message identifies upper variables that need explicit values or, for
    randomized ``best_of`` solving, finite sampling ranges whenever possible.
    Diagnostic notes may include lower-solve or restoration failures.
    """


class SolverUnavailableError(BilevelError, RuntimeError):
    """Raised when a requested numerical backend is unavailable.

    BLVPY translates CVXPY's missing-solver response and native import or
    loading failures into this exception at the actual solve call.
    """


class SolveError(BilevelError, RuntimeError):
    """Raised when a required numerical operation cannot produce a result.

    The main public use is a failed reference lower solve requested by
    :meth:`blvpy.BilevelProblem.gap_diagnostics`. Derivative-compilation and
    restoration failures encountered during ``solve()`` are normally captured
    in run histories or summarized by :class:`blvpy.InitializationError`.
    """
