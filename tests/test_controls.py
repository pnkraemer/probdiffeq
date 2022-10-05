"""Tests for control algorithms."""

import jax.numpy as jnp
import pytest_cases

from odefilter import controls


@pytest_cases.case
def case_proportional_integral():
    return controls.proportional_integral()


@pytest_cases.parametrize_with_cases("control", cases=".")
def test_control(control):
    alg, params = control
    state = alg.init_fn()
    assert state.scale_factor > 0.0

    error_normalised = 10.0
    error_order = 3
    state = alg.step_fn(
        state=state,
        error_normalised=error_normalised,
        error_order=error_order,
        params=params,
    )
    assert state.scale_factor > 0.0
