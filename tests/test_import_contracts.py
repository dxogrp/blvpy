from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from dataclasses import fields

import pytest

import blvpy
import blvpy.cones as cones
import blvpy.continuation as continuation
from blvpy._cones import power


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
