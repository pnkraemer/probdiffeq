"""Tests for control algorithms."""

import pytest_cases

from odefilter import controls


@pytest_cases.parametrize("safety", [0.9, 0.8])
def case_proportional_integral(safety):
    return controls.ProportionalIntegral(safety=safety)


@pytest_cases.parametrize("safety", [0.9, 0.8])
def case_integral(safety):
    return controls.Integral(safety=safety)


@pytest_cases.parametrize_with_cases("control", cases=".")
def test_control(control):

    assert isinstance(control, controls.AbstractControl)

    state = control.init_fn()
    assert state.scale_factor > 0.0

    error_normalised = 10.0
    error_order = 3
    state = control.control_fn(
        state=state,
        error_normalised=error_normalised,
        error_order=error_order,
    )
    assert state.scale_factor > 0.0
