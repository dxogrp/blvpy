from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from dataclasses import fields

import pytest

import blvpy
import blvpy.canonicalization as canonicalization
import blvpy.cones as cones
import blvpy.continuation as continuation
import blvpy.problem as problem_module
from blvpy._canonicalization import affine, audit, parameters, recovery
from blvpy._cones import power
from blvpy._continuation import residuals, restoration, sampling, state


def test_cone_public_objects_remain_defined_by_the_facade() -> None:
    assert blvpy.ConeBlock is cones.ConeBlock
    assert blvpy.ConeLayout is cones.ConeLayout
    assert cones.ConeBlock.__module__ == "blvpy.cones"
    assert cones.ConeLayout.__module__ == "blvpy.cones"
    assert tuple(field.name for field in fields(cones.ConeBlock)) == ("kind", "start", "stop", "index")
    assert tuple(field.name for field in fields(cones.ConeLayout)) == (
        "zero",
        "nonnegative",
        "second_order",
        "power_3d",
        "exponential",
    )


def test_cone_public_signatures_remain_stable() -> None:
    expected = {
        "ConeBlock": "(kind: 'ConeKind', start: 'int', stop: 'int', index: 'int' = 0) -> None",
        "ConeLayout": (
            "(zero: 'int' = 0, nonnegative: 'int' = 0, second_order: 'tuple[int, ...]' = (), "
            "power_3d: 'tuple[float, ...]' = (), exponential: 'int' = 0) -> None"
        ),
        "dual_cone_constraints": (
            "(value: 'cp.Expression | ArrayLike', layout: 'ConeLayout') -> 'tuple[cp.Constraint, ...]'"
        ),
        "dual_cone_distance": "(value: 'ArrayLike', layout: 'ConeLayout') -> 'float'",
        "primal_cone_constraints": (
            "(value: 'cp.Expression | ArrayLike', layout: 'ConeLayout') -> 'tuple[cp.Constraint, ...]'"
        ),
        "primal_cone_distance": "(value: 'ArrayLike', layout: 'ConeLayout') -> 'float'",
        "soc_distance": "(value: 'ArrayLike') -> 'float'",
    }

    assert {name: str(inspect.signature(getattr(cones, name))) for name in expected} == expected


def test_power_cone_dual_scale_remains_one_shared_implementation() -> None:
    assert cones._power_3d_dual_scale is power._power_3d_dual_scale
    assert continuation._power_3d_dual_scale is power._power_3d_dual_scale


def test_canonicalization_public_objects_remain_defined_by_the_facade() -> None:
    expected_fields = {
        "AffineRecoveryMap": ("specs",),
        "CanonicalData": ("A", "b", "c", "d"),
        "CanonicalExpressions": ("A", "b", "c", "d"),
        "CanonicalLowerProblem": (
            "_source_problem",
            "_canonical_problem",
            "_parameter_links",
            "canonical_variable_offsets",
            "cone_layout",
            "canonical_size",
            "constraint_size",
            "parameter_specs",
            "recovery_specs",
            "_affine_map",
            "fixed_parameter_values",
        ),
        "ParameterSpec": (
            "parameter_id",
            "name",
            "shape",
            "size",
            "mapped",
            "internal_parameter_id",
            "internal_shape",
            "internal_size",
            "offset",
            "transform",
            "sparse_indices",
        ),
        "RecoverySpec": ("variable_id", "name", "shape", "matrix", "offset"),
    }

    for name, expected in expected_fields.items():
        public_object = getattr(blvpy, name)
        facade_object = getattr(canonicalization, name)
        assert public_object is facade_object
        assert facade_object.__module__ == "blvpy.canonicalization"
        assert tuple(field.name for field in fields(facade_object)) == expected


def test_canonicalization_public_signatures_remain_stable() -> None:
    expected = {
        "AffineRecoveryMap": "(specs: 'tuple[RecoverySpec, ...]') -> None",
        "CanonicalData": "(A: 'sp.csc_array', b: 'NDArray[np.float64]', c: 'NDArray[np.float64]', d: 'float') -> None",
        "CanonicalExpressions": (
            "(A: 'cp.Expression', b: 'cp.Expression', c: 'cp.Expression', d: 'cp.Expression') -> None"
        ),
        "CanonicalLowerProblem": (
            "(_source_problem: 'cp.Problem', _canonical_problem: 'cp.Problem', "
            "_parameter_links: 'Mapping[cp.Parameter, cp.Expression]', "
            "canonical_variable_offsets: 'Mapping[int, int]', cone_layout: 'ConeLayout', canonical_size: 'int', "
            "constraint_size: 'int', parameter_specs: 'tuple[ParameterSpec, ...]', "
            "recovery_specs: 'tuple[RecoverySpec, ...]', _affine_map: '_DataAffineMap', "
            "fixed_parameter_values: 'Mapping[int, NDArray[np.float64]]') -> None"
        ),
        "ParameterSpec": (
            "(parameter_id: 'int', name: 'str', shape: 'tuple[int, ...]', size: 'int', mapped: 'bool', "
            "internal_parameter_id: 'int', internal_shape: 'tuple[int, ...]', internal_size: 'int', "
            "offset: 'int', transform: 'ParameterTransform' = 'identity', "
            "sparse_indices: 'tuple[tuple[int, ...], ...]' = ()) -> None"
        ),
        "RecoverySpec": (
            "(variable_id: 'int', name: 'str', shape: 'tuple[int, ...]', matrix: 'NDArray[np.float64]', "
            "offset: 'NDArray[np.float64]') -> None"
        ),
        "_canonicalize_lower": (
            "(lower_problem: 'cp.Problem', parameter_links: 'Mapping[cp.Parameter, cp.Expression]') "
            "-> 'CanonicalLowerProblem'"
        ),
        "_validate_lower": (
            "(lower_problem: 'cp.Problem', parameter_links: 'Mapping[cp.Parameter, cp.Expression]') -> 'None'"
        ),
        "_symbolic_matrix_combination": (
            "(coefficients: 'tuple[sp.csc_array, ...]', parameters: 'cp.Expression', rows: 'int', columns: 'int') "
            "-> 'cp.Expression'"
        ),
        "_symbolic_vector_combination": (
            "(coefficients: 'NDArray[np.float64]', parameters: 'cp.Expression') -> 'cp.Expression'"
        ),
    }

    assert {name: str(inspect.signature(getattr(canonicalization, name))) for name in expected} == expected
    assert canonicalization._canonicalize_lower.__module__ == "blvpy.canonicalization"


def test_canonicalization_facade_preserves_private_helper_aliases() -> None:
    expected_aliases = {
        "_DataAffineMap": affine._DataAffineMap,
        "_extract_affine_map": affine._extract_affine_map,
        "_matrix_and_offset": affine._matrix_and_offset,
        "_readonly_vector": affine._readonly_vector,
        "_symbolic_matrix_combination": affine._symbolic_matrix_combination,
        "_symbolic_vector_combination": affine._symbolic_vector_combination,
        "_AUDITED_NONLINEAR_ATOMS": audit._AUDITED_NONLINEAR_ATOMS,
        "_AUDITED_REDUCTION_CHAIN": audit._AUDITED_REDUCTION_CHAIN,
        "_approximation_error": audit._approximation_error,
        "_approximation_metadata": audit._approximation_metadata,
        "_audit_reduction_chain": audit._audit_reduction_chain,
        "_audit_source_atoms": audit._audit_source_atoms,
        "_reject_approximate_source_nodes": audit._reject_approximate_source_nodes,
        "_safe_metadata_repr": audit._safe_metadata_repr,
        "_validate_lower": audit._validate_lower,
        "ParameterTransform": parameters.ParameterTransform,
        "_extract_parameter_specs": parameters._extract_parameter_specs,
        "_freeze_unmapped_parameters": parameters._freeze_unmapped_parameters,
        "_normalise_expression_keys": parameters._normalise_expression_keys,
        "_normalise_value_keys": parameters._normalise_value_keys,
        "_parameter_by_id": parameters._parameter_by_id,
        "_parameter_transform": parameters._parameter_transform,
        "_extract_recovery_specs": recovery._extract_recovery_specs,
        "_recover_source_values": recovery._recover_source_values,
    }

    assert all(getattr(canonicalization, name) is implementation for name, implementation in expected_aliases.items())
    assert problem_module._canonicalize_lower is canonicalization._canonicalize_lower
    assert problem_module._validate_lower is canonicalization._validate_lower
    assert (
        inspect.signature(parameters._extract_parameter_specs).parameters["parameter_spec_factory"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(recovery._extract_recovery_specs).parameters["recovery_spec_factory"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_continuation_facade_preserves_types_signatures_and_exports() -> None:
    expected_signatures = {
        "_SolveSettings": (
            "(epsilon_initial: 'float' = 0.1, epsilon_target: 'float' = 1e-06, contraction: 'float' = 0.1, "
            "best_of: 'int | None' = None, feasibility_tolerance: 'float' = 1e-07, "
            "seed: 'int | np.random.Generator | None' = None, solver: 'str' = 'IPOPT', "
            "conic_solver: 'str' = 'CLARABEL', solver_options: 'Mapping[str, Any] | None' = None, "
            "conic_solver_options: 'Mapping[str, Any] | None' = None, restoration: 'bool' = True, "
            "max_retries: 'int' = 8, verbose: 'bool' = True, solver_verbose: 'bool' = False) -> None"
        ),
        "compute_residuals": "(model: 'BilevelProblem', epsilon: 'float | None' = None) -> 'Residuals'",
        "solve_bilevel": "(model: 'BilevelProblem', settings: '_SolveSettings') -> 'BilevelResult'",
    }

    assert continuation._RunOutcome.__module__ == "blvpy.continuation"
    assert continuation._SolveSettings.__module__ == "blvpy.continuation"
    assert continuation.compute_residuals.__module__ == "blvpy.continuation"
    assert continuation.solve_bilevel.__module__ == "blvpy.continuation"
    assert tuple(field.name for field in fields(continuation._RunOutcome)) == (
        "record",
        "state",
        "accepted_initial",
        "reached_target",
    )
    assert tuple(field.name for field in fields(continuation._SolveSettings)) == (
        "epsilon_initial",
        "epsilon_target",
        "contraction",
        "best_of",
        "feasibility_tolerance",
        "seed",
        "solver",
        "conic_solver",
        "solver_options",
        "conic_solver_options",
        "restoration",
        "max_retries",
        "verbose",
        "solver_verbose",
    )
    assert {name: str(inspect.signature(getattr(continuation, name))) for name in expected_signatures} == (
        expected_signatures
    )
    assert continuation.__all__ == ["compute_residuals"]


def test_continuation_facade_preserves_private_helper_aliases() -> None:
    expected_aliases = {
        "_compute_residuals": residuals.compute_residuals,
        "_constraint_violation": residuals._constraint_violation,
        "_finite_constraint_violation": residuals._finite_constraint_violation,
        "_infinite_residuals": residuals._infinite_residuals,
        "_norm": residuals._norm,
        "_required_vector": residuals._required_vector,
        "_relax_constraint": restoration._relax_constraint,
        "_relaxed_cone_constraints": restoration._relaxed_cone_constraints,
        "_generate_upper_initializations": sampling._generate_upper_initializations,
        "_project_variable_value": sampling._project_variable_value,
        "_validated_sample_bounds": sampling._validated_sample_bounds,
        "_variable_bounds": sampling._variable_bounds,
        "_assign_values": state._assign_values,
        "_numeric_value": state._numeric_value,
        "_restore_state": state._restore_state,
        "_snapshot_state": state._snapshot_state,
        "_sync_linked_parameters": state._sync_linked_parameters,
    }

    assert continuation.compute_residuals is not residuals.compute_residuals
    assert all(getattr(continuation, name) is implementation for name, implementation in expected_aliases.items())


def test_retired_root_private_cone_modules_are_not_importable() -> None:
    assert importlib.util.find_spec("blvpy._cone_numeric") is None
    assert importlib.util.find_spec("blvpy._exponential_cone") is None
    assert importlib.util.find_spec("blvpy._power_cone") is None


@pytest.mark.parametrize(
    "modules",
    [
        (
            "blvpy",
            "blvpy.cones",
            "blvpy.continuation",
            "blvpy._cones.numeric",
            "blvpy._cones.exponential",
            "blvpy._cones.power",
        ),
        (
            "blvpy._cones.numeric",
            "blvpy._cones.exponential",
            "blvpy._cones.power",
            "blvpy.cones",
            "blvpy.continuation",
            "blvpy",
        ),
        (
            "blvpy.continuation",
            "blvpy._cones.power",
            "blvpy.cones",
            "blvpy._cones.exponential",
            "blvpy._cones.numeric",
            "blvpy",
        ),
    ],
    ids=("public-first", "private-first", "continuation-first"),
)
def test_cone_modules_import_cleanly_in_fresh_processes(modules: tuple[str, ...]) -> None:
    script = f"""
import importlib

for module in {modules!r}:
    importlib.import_module(module)

import blvpy
import blvpy.cones as cones
import blvpy.continuation as continuation
from blvpy._cones import power

assert blvpy.ConeBlock is cones.ConeBlock
assert blvpy.ConeLayout is cones.ConeLayout
assert cones._power_3d_dual_scale is power._power_3d_dual_scale
assert continuation._power_3d_dual_scale is power._power_3d_dual_scale
"""
    subprocess.run([sys.executable, "-W", "error", "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "modules",
    [
        (
            "blvpy",
            "blvpy.canonicalization",
            "blvpy.problem",
            "blvpy._canonicalization.affine",
            "blvpy._canonicalization.audit",
            "blvpy._canonicalization.parameters",
            "blvpy._canonicalization.recovery",
        ),
        (
            "blvpy._canonicalization.affine",
            "blvpy._canonicalization.parameters",
            "blvpy._canonicalization.recovery",
            "blvpy._canonicalization.audit",
            "blvpy.canonicalization",
            "blvpy.problem",
            "blvpy",
        ),
        (
            "blvpy.problem",
            "blvpy.canonicalization",
            "blvpy._canonicalization.audit",
            "blvpy._canonicalization.recovery",
            "blvpy._canonicalization.parameters",
            "blvpy._canonicalization.affine",
            "blvpy",
        ),
    ],
    ids=("public-first", "private-first", "problem-first"),
)
def test_canonicalization_modules_import_cleanly_in_fresh_processes(modules: tuple[str, ...]) -> None:
    script = f"""
import importlib

for module in {modules!r}:
    importlib.import_module(module)

import blvpy
import blvpy.canonicalization as canonicalization
import blvpy.problem as problem
from blvpy._canonicalization import affine, audit

assert blvpy.CanonicalData is canonicalization.CanonicalData
assert blvpy.CanonicalLowerProblem is canonicalization.CanonicalLowerProblem
assert canonicalization._canonicalize_lower.__module__ == "blvpy.canonicalization"
assert canonicalization._symbolic_matrix_combination is affine._symbolic_matrix_combination
assert canonicalization._validate_lower is audit._validate_lower
assert problem._canonicalize_lower is canonicalization._canonicalize_lower
assert problem._validate_lower is canonicalization._validate_lower
"""
    subprocess.run([sys.executable, "-W", "error", "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "modules",
    [
        (
            "blvpy",
            "blvpy.continuation",
            "blvpy.problem",
            "blvpy._continuation.state",
            "blvpy._continuation.sampling",
            "blvpy._continuation.residuals",
            "blvpy._continuation.restoration",
        ),
        (
            "blvpy._continuation.state",
            "blvpy._continuation.residuals",
            "blvpy._continuation.sampling",
            "blvpy._continuation.restoration",
            "blvpy.continuation",
            "blvpy.problem",
            "blvpy",
        ),
        (
            "blvpy.problem",
            "blvpy.continuation",
            "blvpy._continuation.restoration",
            "blvpy._continuation.sampling",
            "blvpy._continuation.residuals",
            "blvpy._continuation.state",
            "blvpy",
        ),
    ],
    ids=("public-first", "private-first", "problem-first"),
)
def test_continuation_modules_import_cleanly_in_fresh_processes(modules: tuple[str, ...]) -> None:
    script = f"""
import importlib

for module in {modules!r}:
    importlib.import_module(module)

import blvpy.continuation as continuation
from blvpy._continuation import residuals, restoration, sampling, state

assert continuation._SolveSettings.__module__ == "blvpy.continuation"
assert continuation.compute_residuals.__module__ == "blvpy.continuation"
assert continuation.__all__ == ["compute_residuals"]
assert continuation._compute_residuals is residuals.compute_residuals
assert continuation._generate_upper_initializations is sampling._generate_upper_initializations
assert continuation._relaxed_cone_constraints is restoration._relaxed_cone_constraints
assert continuation._snapshot_state is state._snapshot_state
"""
    subprocess.run([sys.executable, "-W", "error", "-c", script], check=True, capture_output=True, text=True)
