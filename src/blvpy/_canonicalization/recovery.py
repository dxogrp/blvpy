"""Affine recovery extraction for source lower variables."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import cvxpy as cp
import numpy as np
from cvxpy import settings as cvxpy_settings
from cvxpy.reductions.solution import Solution
from numpy.typing import NDArray

from ..errors import CanonicalizationError

_RecoverySpecT = TypeVar("_RecoverySpecT")


def _extract_recovery_specs(
    problem: cp.Problem,
    chain: Any,
    inverse_data: list[Any],
    param_prog: Any,
    canonical_size: int,
    *,
    recovery_spec_factory: Callable[..., _RecoverySpecT],
) -> tuple[_RecoverySpecT, ...]:
    zero = _recover_source_values(np.zeros(canonical_size), problem, chain, inverse_data, param_prog)
    matrices = {variable.id: np.empty((variable.size, canonical_size), dtype=float) for variable in problem.variables()}
    for column in range(canonical_size):
        basis = np.zeros(canonical_size)
        basis[column] = 1.0
        recovered = _recover_source_values(basis, problem, chain, inverse_data, param_prog)
        for variable in problem.variables():
            matrices[variable.id][:, column] = np.asarray(recovered[variable.id]).reshape(-1, order="F") - np.asarray(
                zero[variable.id]
            ).reshape(-1, order="F")

    specs: list[_RecoverySpecT] = []
    for variable in problem.variables():
        if variable.id not in zero:
            raise CanonicalizationError(
                f"CVXPY did not provide recovery metadata for lower variable {variable.name()!r}."
            )
        specs.append(
            recovery_spec_factory(
                variable_id=int(variable.id),
                name=variable.name() or f"var_{variable.id}",
                shape=tuple(int(size) for size in variable.shape),
                matrix=matrices[variable.id],
                offset=np.asarray(zero[variable.id]).reshape(-1, order="F"),
            )
        )
    return tuple(specs)


def _recover_source_values(
    canonical_value: NDArray[np.float64],
    problem: cp.Problem,
    chain: Any,
    inverse_data: list[Any],
    param_prog: Any,
) -> dict[int, NDArray[np.float64]]:
    dual_values = {constraint.id: np.zeros(constraint.shape, dtype=float) for constraint in param_prog.constraints}
    solution = Solution(
        cvxpy_settings.OPTIMAL,
        0.0,
        {param_prog.x.id: canonical_value},
        dual_values,
        {},
    )
    try:
        # Skip the solver interface. Its invert step expects a Clarabel-native
        # object and has no role in the source-variable affine map.
        for reduction, inverse in reversed(list(zip(chain.reductions[:-1], inverse_data[:-1]))):
            # CvxAttr2Constr.invert projects ordinary sign/bound-attributed
            # variables. Projection is harmless for a feasible solver point
            # but nonlinear, so it cannot define the fixed affine recovery map
            # required by DBLP. Its public ``var_forward`` performs exactly the
            # desired linear unpacking, including symmetric/sparse variables.
            if type(reduction).__name__ == "CvxAttr2Constr":
                solution = Solution(
                    solution.status,
                    solution.opt_val,
                    reduction.var_forward(solution.primal_vars),
                    solution.dual_vars,
                    solution.attr,
                )
            else:
                solution = reduction.invert(solution, inverse)
    except Exception as error:
        raise CanonicalizationError("CVXPY source-variable recovery failed.") from error
    result: dict[int, NDArray[np.float64]] = {}
    for variable in problem.variables():
        value = solution.primal_vars.get(variable.id)
        if value is not None:
            result[variable.id] = np.asarray(value, dtype=float)
    return result
