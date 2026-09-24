"""Source-model validation and exact-canonicalization audits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.cummax import cummax
from cvxpy.atoms.dotsort import dotsort
from cvxpy.atoms.elementwise.abs import abs as abs_atom
from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.huber import huber
from cvxpy.atoms.elementwise.kl_div import kl_div
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.log1p import log1p
from cvxpy.atoms.elementwise.logistic import logistic
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.power import Power, PowerApprox
from cvxpy.atoms.elementwise.rel_entr import rel_entr
from cvxpy.atoms.elementwise.xexp import xexp
from cvxpy.atoms.geo_mean import GeoMeanApprox
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.max import max as max_atom
from cvxpy.atoms.min import min as min_atom
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.pnorm import Pnorm, PnormApprox
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.constraints.power import PowCone3DApprox

from ..errors import (
    ApproximateCanonicalizationError,
    CanonicalizationError,
    ParameterMappingError,
    UnsupportedModelError,
    ValidationError,
)
from .parameters import _freeze_unmapped_parameters

# This is intentionally narrower than “anything CVXPY can turn into a cone program”.
# Each nonlinear entry below has an exact epigraph/hypograph graph whose
# pointwise projection is preserved by CVXPY's Dcp2Cone reduction.  Affine
# atoms are audited as a class because their graph and recovery are identities.
_AUDITED_NONLINEAR_ATOMS = frozenset(
    {
        abs_atom,
        cummax,
        dotsort,
        entr,
        exp,
        GeoMeanApprox,
        huber,
        kl_div,
        log,
        log1p,
        logistic,
        log_sum_exp,
        max_atom,
        maximum,
        min_atom,
        minimum,
        norm1,
        norm_inf,
        Pnorm,
        PnormApprox,
        Power,
        PowerApprox,
        QuadForm,
        quad_over_lin,
        rel_entr,
        sum_largest,
        xexp,
    }
)
_AUDITED_REDUCTION_CHAIN = (
    "Dcp2Cone",
    "CvxAttr2Constr",
    "EliminateZeroSized",
    "ConeMatrixStuffing",
    "CLARABEL",
)


def _validate_lower(
    lower_problem: cp.Problem,
    parameter_links: Mapping[cp.Parameter, cp.Expression],
) -> None:
    """Validate the source-level requirements of a supported conic lower model."""

    if not isinstance(lower_problem, cp.Problem):
        raise ValidationError("lower_problem must be a cvxpy.Problem.")
    if not isinstance(lower_problem.objective, (cp.Minimize, cp.Maximize)):
        raise ValidationError("The lower objective must use cp.Minimize or cp.Maximize.")
    if not lower_problem.objective.expr.is_scalar():
        raise ValidationError("The lower objective must be a scalar expression.")
    if not lower_problem.objective.expr.is_real():
        raise UnsupportedModelError("The lower objective must be real-valued.")
    if lower_problem.is_mixed_integer():
        raise UnsupportedModelError("Mixed-integer lower problems are not supported.")
    for variable in lower_problem.variables():
        if variable.is_complex():
            raise UnsupportedModelError(f"Complex lower variable {variable.name()!r} is not supported.")

    lower_parameters = {parameter.id: parameter for parameter in lower_problem.parameters()}
    seen: set[int] = set()
    try:
        items = tuple(parameter_links.items())
    except AttributeError as error:
        raise ParameterMappingError("The linked-parameter data must be a mapping.") from error
    for parameter, linked_expression in items:
        if not isinstance(parameter, cp.Parameter):
            raise ParameterMappingError("Every linked-parameter key must be a cvxpy.Parameter.")
        if parameter.id not in lower_parameters:
            raise ParameterMappingError(f"Mapped parameter {parameter.name()!r} does not occur in the lower problem.")
        if parameter.id in seen:
            raise ParameterMappingError(f"Parameter {parameter.name()!r} is mapped more than once.")
        seen.add(parameter.id)
        try:
            expression = cp.Expression.cast_to_const(linked_expression)
        except Exception as error:
            raise ParameterMappingError(
                f"Value linked to parameter {parameter.name()!r} is not a CVXPY expression."
            ) from error
        if expression.shape != parameter.shape:
            raise ParameterMappingError(
                f"Parameter {parameter.name()!r} has shape {parameter.shape}, but its linked "
                f"expression has shape {expression.shape}."
            )
        if expression.is_complex():
            raise ParameterMappingError(f"Expression linked to parameter {parameter.name()!r} must be real.")
        if not expression.is_affine():
            raise ParameterMappingError(f"Expression linked to parameter {parameter.name()!r} must be affine.")

    for parameter in lower_problem.parameters():
        if parameter.id not in seen and parameter.value is None:
            raise ParameterMappingError(f"Unmapped lower parameter {parameter.name()!r} must have a fixed value.")

    canonical_problem = _freeze_unmapped_parameters(lower_problem, seen)
    if not canonical_problem.is_dcp():
        raise ValidationError("The lower problem must satisfy CVXPY's DCP rules.")
    if not canonical_problem.is_dpp():
        raise ValidationError("The lower problem must satisfy CVXPY's DPP rules with respect to the mapped parameters.")

    _reject_approximate_source_nodes(lower_problem)


def _reject_approximate_source_nodes(problem: cp.Problem) -> None:
    expressions = [problem.objective.expr]
    expressions.extend(argument for constraint in problem.constraints for argument in constraint.args)
    seen: set[int] = set()
    stack = list(expressions)
    while stack:
        expression = stack.pop()
        if id(expression) in seen:
            continue
        seen.add(id(expression))
        if isinstance(expression, (PowerApprox, PnormApprox, GeoMeanApprox)):
            approximation_error = _approximation_error(expression)
            if not np.isfinite(approximation_error) or approximation_error != 0.0:
                metadata = _approximation_metadata(expression)
                details = f", {metadata}" if metadata else ""
                raise ApproximateCanonicalizationError(
                    f"Atom {type(expression).__name__} has nonzero or nonfinite approximation error "
                    f"(approx_error={approximation_error!r}{details}); "
                    "exact source atoms are required."
                )
        stack.extend(getattr(expression, "args", ()))
    for constraint in problem.constraints:
        if isinstance(constraint, PowCone3DApprox):
            raise ApproximateCanonicalizationError(
                f"Constraint {type(constraint).__name__} uses an SOC approximation of a 3D power cone."
            )
        if isinstance(constraint, (RelEntrConeQuad, OpRelEntrConeQuad)):
            raise ApproximateCanonicalizationError(
                f"Constraint {type(constraint).__name__} uses quadrature approximation."
            )


def _approximation_error(expression: Atom) -> float:
    """Read CVXPY's approximation error without assuming optional metadata."""

    value = getattr(expression, "approx_error", None)
    try:
        array = np.asarray(value, dtype=float)
        if array.shape:
            raise ValueError("approximation error is not scalar")
        return float(array)
    except (TypeError, ValueError, OverflowError) as error:
        raise ApproximateCanonicalizationError(
            f"Atom {type(expression).__name__} has invalid approximation error metadata "
            f"(approx_error={_safe_metadata_repr(value)}); exact source atoms are required."
        ) from error


def _approximation_metadata(expression: Atom) -> str:
    """Return concise atom metadata when CVXPY exposes it safely."""

    if isinstance(expression, PowerApprox):
        metadata = (
            ("requested_p", getattr(expression, "_p_orig", None)),
            ("used_p", getattr(expression, "p_used", getattr(expression, "p", None))),
        )
    elif isinstance(expression, PnormApprox):
        metadata = (
            ("requested_p", getattr(expression, "original_p", None)),
            ("used_p", getattr(expression, "p", None)),
        )
    elif isinstance(expression, GeoMeanApprox):
        metadata = (
            ("requested_weights", getattr(expression, "p", None)),
            ("used_weights", getattr(expression, "w", None)),
        )
    else:
        metadata = ()

    fields: list[str] = []
    for name, value in metadata:
        try:
            if isinstance(value, cp.Expression):
                value = value.value if value.is_constant() else None
        except Exception:
            continue
        if value is not None:
            fields.append(f"{name}={_safe_metadata_repr(value)}")
    return ", ".join(fields)


def _safe_metadata_repr(value: Any) -> str:
    """Bound diagnostics for metadata whose concrete CVXPY type may vary."""

    try:
        if isinstance(value, np.ndarray):
            rendered = np.array2string(value, threshold=8, edgeitems=2, max_line_width=80)
        else:
            rendered = repr(value)
    except Exception:
        return "<unavailable>"
    if len(rendered) > 160:
        return rendered[:157] + "..."
    return rendered


def _audit_source_atoms(problem: cp.Problem) -> None:
    """Enforce the explicit pointwise-graph atom allowlist for conic mode."""

    expressions = [problem.objective.expr]
    expressions.extend(argument for constraint in problem.constraints for argument in constraint.args)
    seen: set[int] = set()
    stack = list(expressions)
    while stack:
        expression = stack.pop()
        if id(expression) in seen:
            continue
        seen.add(id(expression))
        if isinstance(expression, Atom) and not isinstance(expression, AffAtom):
            atom_type = type(expression)
            if atom_type not in _AUDITED_NONLINEAR_ATOMS:
                raise UnsupportedModelError(
                    f"Atom {atom_type.__name__} is not in BLVPY's audited exact conic canonicalization allowlist."
                )
        stack.extend(getattr(expression, "args", ()))


def _audit_reduction_chain(chain: Any) -> None:
    names = tuple(type(reduction).__name__ for reduction in chain.reductions)
    if names != _AUDITED_REDUCTION_CHAIN and names != (
        "Dcp2Cone",
        "CvxAttr2Constr",
        "ExactCone2Cone",
        "EliminateZeroSized",
        "ConeMatrixStuffing",
        "CLARABEL",
    ):
        raise CanonicalizationError("CVXPY selected an unaudited canonicalization chain: " + " -> ".join(names))
