"""Affine canonical-data extraction and symbolic combination helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class _DataAffineMap:
    """Coefficients of canonical data against CVXPY's packed parameter vector."""

    A: tuple[sp.csc_array, ...]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    d: NDArray[np.float64]


def _extract_affine_map(param_prog: Any, rows: int, columns: int) -> _DataAffineMap:
    parameter_vector_size = int(param_prog.total_param_size) + 1
    A_coefficients: list[sp.csc_array] = []
    b_coefficients = np.empty((rows, parameter_vector_size), dtype=float)
    c_coefficients = np.empty((columns, parameter_vector_size), dtype=float)
    d_coefficients = np.empty(parameter_vector_size, dtype=float)
    param_prog.reduced_A.cache(True)
    for index in range(parameter_vector_size):
        basis = np.zeros(parameter_vector_size, dtype=float)
        basis[index] = 1.0
        c_sparse, d = _matrix_and_offset(param_prog.q, basis, columns)
        A, b = param_prog.reduced_A.get_matrix_from_tensor(basis, with_offset=True)
        # ConeMatrixStuffing stores the affine constraint expression F @ u + g
        # while conic solver interfaces expose ``A=-F`` and ``b=g``.  BLVPY's
        # public convention follows the latter: A @ u + s == b.
        A_coefficients.append(-sp.csc_array(A, dtype=float))
        b_coefficients[:, index] = np.asarray(b, dtype=float).reshape(-1)
        c_coefficients[:, index] = np.asarray(c_sparse.toarray(), dtype=float).reshape(-1)
        d_coefficients[index] = float(np.asarray(d).reshape(()))
    for array in (b_coefficients, c_coefficients, d_coefficients):
        array.setflags(write=False)
    return _DataAffineMap(
        A=tuple(A_coefficients),
        b=b_coefficients,
        c=c_coefficients,
        d=d_coefficients,
    )


def _matrix_and_offset(tensor: Any, parameter_vector: NDArray[np.float64], length: int) -> tuple[Any, Any]:
    # This is CVXPY's stable tensor contract used by ParamConeProg itself.
    from cvxpy.cvxcore.python import canonInterface

    return canonInterface.get_matrix_from_tensor(tensor, parameter_vector, length, with_offset=True)


def _symbolic_matrix_combination(
    coefficients: tuple[sp.csc_array, ...],
    parameters: cp.Expression,
    rows: int,
    columns: int,
) -> cp.Expression:
    """Apply sparse affine-matrix coefficients to packed parameters."""

    if rows == 0 or columns == 0:
        return cp.Constant(np.empty((rows, columns)))
    operator = sp.hstack(
        [coefficient.reshape((-1, 1), order="C") for coefficient in coefficients],
        format="csc",
    )
    vector = cp.Constant(operator) @ parameters
    return cp.reshape(vector, (rows, columns), order="C")


def _symbolic_vector_combination(
    coefficients: NDArray[np.float64],
    parameters: cp.Expression,
) -> cp.Expression:
    """Apply sparse affine-vector coefficients to packed parameters."""

    if coefficients.shape[0] == 0:
        return cp.Constant(np.empty(0))
    return cp.Constant(sp.csc_array(coefficients)) @ parameters


def _readonly_vector(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float).reshape(-1).copy()
    array.setflags(write=False)
    return array
