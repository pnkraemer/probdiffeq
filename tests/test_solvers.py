"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import backends, controls, information, inits, odefilters, solvers


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_isotropic_filter(num_derivatives, information_op):
    return backends.DynamicFilter(
        implementation=backends.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_isotropic_smoother(num_derivatives, information_op):
    return backends.DynamicSmoother(
        implementation=backends.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize("information_op", [information.EK1(ode_dimension=1)])
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_ek1_filter(num_derivatives, information_op):
    return backends.DynamicFilter(
        implementation=backends.DenseImplementation.from_num_derivatives(
            num_derivatives=num_derivatives, ode_dimension=1
        ),
        information=information_op,
    )


@pytest_cases.parametrize("derivative_init_fn", [inits.taylor_mode, inits.forward_mode])
@pytest_cases.parametrize_with_cases("backend", cases=".", prefix="case_backend_")
def case_solver_adaptive_ek0(derivative_init_fn, backend):
    odefilter = odefilters.ODEFilter(
        derivative_init_fn=derivative_init_fn,
        backend=backend,
    )
    control = controls.ProportionalIntegral()
    atol, rtol = 1e-3, 1e-3
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=backend.implementation.num_derivatives + 1,
    )


def case_ivp_logistic():
    def vf(x, t):
        return x * (1 - x)

    return vf, (jnp.asarray([0.4]),), 0.0, 2.0


@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
@pytest_cases.parametrize_with_cases("vf, u0, t0, t1", cases=".", prefix="case_ivp_")
def test_solver(solver, vf, u0, t0, t1):
    assert isinstance(solver, solvers.AbstractIVPSolver)

    state0 = solver.init_fn(
        vector_field=vf,
        initial_values=u0,
        t0=t0,
    )
    assert state0.dt_proposed > 0.0
    assert state0.accepted.t == t0
    assert jnp.shape(state0.accepted.u) == jnp.shape(u0[0])

    state1 = solver.step_fn(state=state0, vector_field=vf, t1=t1)
    assert isinstance(state0, type(state1))
    assert state1.dt_proposed > 0.0
    assert t0 < state1.accepted.t <= t1
    assert jnp.shape(state1.proposed.u) == jnp.shape(u0[0])
