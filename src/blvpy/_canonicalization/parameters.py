"""Parameter freezing, packing metadata, and key normalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar

import cvxpy as cp
from numpy.typing import ArrayLike

from ..errors import CanonicalizationError

ParameterTransform = Literal["identity", "symmetric", "diagonal", "sparse"]

_ParameterSpecT = TypeVar("_ParameterSpecT")


def _freeze_unmapped_parameters(problem: cp.Problem, mapped_ids: set[int]) -> cp.Problem:
    """Replace fixed-data parameters by constants before DPP canonicalization."""

    replacements = {
        parameter.id: cp.Constant(parameter.value)
        for parameter in problem.parameters()
        if parameter.id not in mapped_ids
    }
    if not replacements:
        return problem

    def replace(expression: cp.Expression) -> cp.Expression:
        if isinstance(expression, cp.Parameter) and expression.id in replacements:
            return replacements[expression.id]
        if not any(parameter.id in replacements for parameter in expression.parameters()):
            return expression
        new_args = [replace(argument) for argument in expression.args]
        return expression.copy(new_args)

    objective = type(problem.objective)(replace(problem.objective.expr))
    constraints: list[cp.Constraint] = []
    for constraint in problem.constraints:
        arguments = [replace(argument) for argument in constraint.args]
        data = constraint.get_data()
        constraints.append(type(constraint)(*(arguments + data)) if data is not None else type(constraint)(*arguments))
    return cp.Problem(objective, constraints)


def _extract_parameter_specs(
    problem: cp.Problem,
    mapping: Mapping[cp.Parameter, cp.Expression],
    chain: Any,
    param_prog: Any,
    *,
    parameter_spec_factory: Callable[..., _ParameterSpecT],
) -> tuple[_ParameterSpecT, ...]:
    mapped_ids = {parameter.id for parameter in mapping}
    internal_by_original: dict[int, tuple[cp.Parameter, ParameterTransform]] = {}
    attr_reduction = next(
        (reduction for reduction in chain.reductions if type(reduction).__name__ == "CvxAttr2Constr"),
        None,
    )
    replaced = getattr(attr_reduction, "_parameters", {}) if attr_reduction is not None else {}
    for parameter in problem.parameters():
        internal = replaced.get(parameter, parameter)
        internal_by_original[parameter.id] = (internal, _parameter_transform(parameter, internal))

    specs: list[_ParameterSpecT] = []
    for parameter in problem.parameters():
        internal, transform = internal_by_original[parameter.id]
        if internal.id not in param_prog.param_id_to_col:
            raise CanonicalizationError(f"CVXPY's canonical parameter vector omits {parameter.name()!r}.")
        sparse_indices: tuple[tuple[int, ...], ...] = ()
        if transform == "sparse":
            sparse_indices = tuple(tuple(int(value) for value in axis) for axis in parameter.sparse_idx)
        specs.append(
            parameter_spec_factory(
                parameter_id=int(parameter.id),
                name=parameter.name() or f"param_{parameter.id}",
                shape=tuple(int(size) for size in parameter.shape),
                size=int(parameter.size),
                mapped=parameter.id in mapped_ids,
                internal_parameter_id=int(internal.id),
                internal_shape=tuple(int(size) for size in internal.shape),
                internal_size=int(internal.size),
                offset=int(param_prog.param_id_to_col[internal.id]),
                transform=transform,
                sparse_indices=sparse_indices,
            )
        )
    return tuple(specs)


def _parameter_transform(parameter: cp.Parameter, internal: cp.Parameter) -> ParameterTransform:
    if parameter is internal:
        return "identity"
    attributes = parameter.attributes
    if any(attributes.get(name, False) for name in ("symmetric", "PSD", "NSD")):
        return "symmetric"
    if attributes.get("diag", False):
        return "diagonal"
    if attributes.get("sparsity", False):
        return "sparse"
    raise CanonicalizationError(f"Unsupported parameter dimension reduction for {parameter.name()!r}.")


def _normalise_value_keys(
    values: Mapping[cp.Parameter | int, ArrayLike],
) -> dict[int, ArrayLike]:
    result: dict[int, ArrayLike] = {}
    for key, value in values.items():
        parameter_id = int(key.id) if isinstance(key, cp.Parameter) else int(key)
        result[parameter_id] = value
    return result


def _normalise_expression_keys(
    values: Mapping[cp.Parameter | int, cp.Expression],
) -> dict[int, cp.Expression]:
    result: dict[int, cp.Expression] = {}
    for key, value in values.items():
        parameter_id = int(key.id) if isinstance(key, cp.Parameter) else int(key)
        result[parameter_id] = cp.Expression.cast_to_const(value)
    return result


def _parameter_by_id(problem: cp.Problem, parameter_id: int) -> cp.Parameter:
    for parameter in problem.parameters():
        if parameter.id == parameter_id:
            return parameter
    raise CanonicalizationError(f"Unknown lower parameter ID {parameter_id}.")
