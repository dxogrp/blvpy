"""Public exception hierarchy for BLVPY."""

from __future__ import annotations


class BilevelError(Exception):
    """Base class for errors raised by BLVPY."""


class ValidationError(BilevelError, ValueError):
    """The supplied model is not a supported disciplined bilevel model."""


class ParameterMappingError(ValidationError):
    """A lower parameter cannot be linked to the supplied upper expression."""


class UnsupportedModelError(ValidationError):
    """The lower model uses a feature outside BLVPY's supported subset."""


class UnsupportedConeError(UnsupportedModelError):
    """Canonicalization produced a cone not supported by this release."""


class ApproximateCanonicalizationError(UnsupportedModelError):
    """An atom or constraint would be represented only approximately."""


class CanonicalizationError(BilevelError, RuntimeError):
    """CVXPY failed to produce or expose the expected canonical program."""


class InitializationError(BilevelError, RuntimeError):
    """No suitable initial point could be constructed."""


class SolverUnavailableError(BilevelError, RuntimeError):
    """A requested numerical backend is not installed or importable."""


class SolveError(BilevelError, RuntimeError):
    """The lifted nonlinear solve failed."""
