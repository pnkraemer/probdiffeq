"""Tests for control algorithms."""

import pytest_cases

from odefilter import controls


@pytest_cases.parametrize("safety", [0.9, 0.8])
def case_proportional_integral(safety):
    return controls.proportional_integral(safety=safety)


@pytest_cases.parametrize("safety", [0.9, 0.8])
def case_integral(safety):
    return controls.integral(safety=safety)


@pytest_cases.parametrize_with_cases("control", cases=".")
def test_control(control):
    alg, params = control

    assert isinstance(alg, controls.AbstractControl)

    state = alg.init_fn()
    assert state.scale_factor > 0.0

    error_normalised = 10.0
    error_order = 3
    state = alg.control_fn(
        state=state,
        error_normalised=error_normalised,
        error_order=error_order,
        params=params,
    )
    assert state.scale_factor > 0.0
