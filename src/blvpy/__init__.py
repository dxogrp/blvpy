"""BLVpy: disciplined optimistic bilevel optimization with CVXPY."""

from .canonicalization import (
    AffineRecoveryMap,
    CanonicalData,
    CanonicalExpressions,
    CanonicalLowerProblem,
    ParameterSpec,
    RecoverySpec,
    canonicalize_lower,
    validate_lower,
)
from .cones import ConeBlock, ConeLayout
from .continuation import SolveSettings
from .errors import (
    ApproximateCanonicalizationError,
    BilevelError,
    CanonicalizationError,
    InitializationError,
    ParameterMappingError,
    SolveError,
    SolverUnavailableError,
    UnsupportedConeError,
    UnsupportedModelError,
    ValidationError,
)
from .problem import BilevelProblem, LiftedProblem
from .result import BilevelResult, GapDiagnostics, IterationRecord, Residuals, StartRecord

__all__ = [
    "ApproximateCanonicalizationError",
    "AffineRecoveryMap",
    "BilevelError",
    "BilevelProblem",
    "BilevelResult",
    "CanonicalData",
    "CanonicalExpressions",
    "CanonicalLowerProblem",
    "CanonicalizationError",
    "ConeBlock",
    "ConeLayout",
    "GapDiagnostics",
    "InitializationError",
    "IterationRecord",
    "LiftedProblem",
    "ParameterMappingError",
    "ParameterSpec",
    "RecoverySpec",
    "Residuals",
    "SolveError",
    "SolveSettings",
    "SolverUnavailableError",
    "StartRecord",
    "UnsupportedConeError",
    "UnsupportedModelError",
    "ValidationError",
    "canonicalize_lower",
    "validate_lower",
]

__version__ = "0.1.0"
