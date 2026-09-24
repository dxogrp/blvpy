from __future__ import annotations

import subprocess
import sys
from dataclasses import fields

import pytest

import blvpy
import blvpy.canonicalization as canonicalization
import blvpy.cones as cones
import blvpy.continuation as continuation


def test_documented_public_types_remain_defined_by_their_facades() -> None:
    expected = {
        cones: ("ConeBlock", "ConeLayout"),
        canonicalization: (
            "AffineRecoveryMap",
            "CanonicalData",
            "CanonicalExpressions",
            "CanonicalLowerProblem",
            "ParameterSpec",
            "RecoverySpec",
        ),
    }

    for facade, names in expected.items():
        for name in names:
            public_object = getattr(blvpy, name)
            facade_object = getattr(facade, name)
            assert public_object is facade_object, name
            assert facade_object.__module__ == facade.__name__, name


def test_public_cone_records_preserve_positional_field_order() -> None:
    assert tuple(field.name for field in fields(cones.ConeBlock)) == (
        "kind",
        "start",
        "stop",
        "index",
    )
    assert tuple(field.name for field in fields(cones.ConeLayout)) == (
        "zero",
        "nonnegative",
        "second_order",
        "power_3d",
        "exponential",
    )


def test_power_cone_dual_scale_compatibility_alias_preserves_behavior() -> None:
    alpha = 0.25
    complement = 1.0 - alpha
    expected = alpha**alpha * complement**complement

    assert cones._power_3d_dual_scale(alpha) == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_compute_residuals_remains_a_public_continuation_wrapper() -> None:
    assert continuation.compute_residuals.__module__ == "blvpy.continuation"
    assert "compute_residuals" in continuation.__all__


def test_continuation_private_modules_import_before_facade_in_fresh_process() -> None:
    script = """
import importlib

for module in (
    "blvpy._continuation.state",
    "blvpy._continuation.residuals",
    "blvpy._continuation.sampling",
    "blvpy._continuation.restoration",
    "blvpy.continuation",
):
    importlib.import_module(module)

import blvpy.continuation as continuation

assert continuation.compute_residuals.__module__ == "blvpy.continuation"
assert "compute_residuals" in continuation.__all__
"""
    subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
