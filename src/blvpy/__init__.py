"""BLVPY: disciplined optimistic bilevel optimization with CVXPY."""

from .canonicalization import (
    AffineRecoveryMap,
    CanonicalData,
    CanonicalExpressions,
    CanonicalLowerProblem,
    ParameterSpec,
    RecoverySpec,
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
from .lower_problem import LowerProblem
from .problem import BilevelProblem, LiftedProblem
from .result import BilevelResult, GapDiagnostics, IterationRecord, Residuals, RunRecord

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
    "LowerProblem",
    "ParameterMappingError",
    "ParameterSpec",
    "RecoverySpec",
    "Residuals",
    "RunRecord",
    "SolveError",
    "SolveSettings",
    "SolverUnavailableError",
    "UnsupportedConeError",
    "UnsupportedModelError",
    "ValidationError",
]

__version__ = "0.1.0"
