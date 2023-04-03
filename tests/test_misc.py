"""Tests for miscellaneous edge cases.

Place all tests that have no better place here.
"""
import jax
import jax.numpy as jnp
import pytest
import pytest_cases
from diffeqzoo import backend, ivps

from probdiffeq import controls, ivpsolve, ivpsolvers, test_util
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters


@pytest_cases.parametrize("incr", [1, -1])
@pytest_cases.parametrize("n", [2])
def test_incorrect_number_of_taylor_coefficients_init(incr, n):
    """Assert that a specific ValueError is raised.

    Specifically:
    A ValueError must be raised if the number of Taylor coefficients
    passed to *IBM.init_hidden_state() does not match the `num_derivatives`
    attribute of the extrapolation model.
    """
    solver = test_util.generate_solver(num_derivatives=n)
    tcoeffs_wrong_length = [None] * (n + 1 + incr)  # 'None' bc. values irrelevant

    init_fn = solver.strategy.implementation.extrapolation.init_hidden_state
    with pytest.raises(ValueError):
        init_fn(taylor_coefficients=tcoeffs_wrong_length)


def test_float32_compatibility():
    """Solve three-body problem. Terminal value must not be nan."""

    if not backend.has_been_selected:
        backend.select("jax")  # ivp examples in jax

    f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()

    @jax.jit
    def vf_1(y, t, p):
        return f(y, *p)

    f, (u0, du0), (t0, t1), f_args = ivps.three_body_restricted()

    @jax.jit
    def vf_2(y, dy, t, p):
        return f(y, dy, *p)

    # One derivative more than above because we don't transform to first order
    implementation = recipes.IsoTS0.from_params(ode_order=2, num_derivatives=5)
    ts0_2 = ivpsolvers.CalibrationFreeSolver(
        filters.Filter(implementation), output_scale_sqrtm=1.0
    )
    ts = jnp.linspace(t0, t1, endpoint=True, num=5)

    print("Hypothesis: compiling implies far fewer steps, and somehow NaNs. Why?")

    with jax.disable_jit():
        solution = ivpsolve.solve_with_python_while_loop(
            vf_2,
            initial_values=(u0, du0),
            t0=t0,
            t1=t1,
            solver=ts0_2,
            atol=1e-4,
            rtol=1e-4,
            parameters=f_args,
            control=controls.Integral(),
            numerical_zero=1e-10,
        )
    print(len(solution))
    print(jnp.diff(solution.t))
    # print(solution.u[-10:])
    print(solution.error_estimate[-10:])
    assert not jnp.any(jnp.isnan(solution.u))
