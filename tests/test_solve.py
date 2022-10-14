"""Tests for IVP solvers."""
import jax.numpy as jnp
from jax.experimental.ode import odeint
from pytest_cases import parametrize_with_cases

from odefilter import ivpsolve


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver, info_op", cases=".recipe_cases", prefix="solver_", has_tag=("solve",)
)
def test_solve(vf, u0, t0, t1, p, solver, info_op):
    ts = jnp.linspace(t0, t1, num=10)
    odeint_solution = odeint(
        lambda y, t, *p: vf(t, y, *p), u0[0], ts, *p, atol=1e-6, rtol=1e-6
    )
    ts_reference, ys_reference = ts, odeint_solution

    solution = ivpsolve.solve(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=solver, info_op=info_op
    )
    assert jnp.allclose(solution.t[-1], ts_reference[-1])
    assert jnp.allclose(solution.u[-1], ys_reference[-1], atol=1e-3, rtol=1e-3)
