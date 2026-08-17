"""BLVPY: disciplined optimistic bilevel optimization with CVXPY."""

from importlib.metadata import version as _distribution_version

from .canonicalization import (
    AffineRecoveryMap,
    CanonicalData,
    CanonicalExpressions,
    CanonicalLowerProblem,
    ParameterSpec,
    RecoverySpec,
)
from .cones import ConeBlock, ConeLayout
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
from .problem import BilevelProblem
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
    "LowerProblem",
    "ParameterMappingError",
    "ParameterSpec",
    "RecoverySpec",
    "Residuals",
    "RunRecord",
    "SolveError",
    "SolverUnavailableError",
    "UnsupportedConeError",
    "UnsupportedModelError",
    "ValidationError",
]

__version__ = _distribution_version("blvpy")
