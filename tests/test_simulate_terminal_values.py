"""Tests for IVP solvers."""

import jax.numpy as jnp
from jax.experimental.ode import odeint
from pytest_cases import parametrize_with_cases

from odefilter import ivpsolve


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver",
    cases=".recipe_cases",
    prefix="solver_",
    has_tag=("terminal_value",),
)
def test_simulate_terminal_values(vf, u0, t0, t1, p, solver):
    def func(y, t, *p):
        return vf(y, t=t, p=p)

    odeint_solution = odeint(
        func, u0[0], jnp.asarray([t0, t1]), *p, atol=1e-6, rtol=1e-6
    )
    ys_reference = odeint_solution[-1, :]

    solution = ivpsolve.simulate_terminal_values(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=solver
    )

    assert solution.t == t1
    assert jnp.allclose(solution.u, ys_reference, atol=1e-3, rtol=1e-3)
