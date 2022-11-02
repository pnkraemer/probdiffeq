"""Tests for IVP solvers."""
import jax.numpy as jnp
from jax.experimental.ode import odeint
from pytest_cases import parametrize_with_cases

from odefilter import ivpsolve


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver", cases=".solver_cases", prefix="solver_", has_tag=("solve",)
)
def test_solve_fixed_grid(vf, u0, t0, t1, p, solver):
    ts = jnp.linspace(t0, t1, num=10)
    odeint_solution = odeint(
        lambda y, t, *par: vf(y, t=t, p=par), u0[0], ts, *p, atol=1e-6, rtol=1e-6
    )
    ts_reference, ys_reference = ts, odeint_solution

    solution = ivpsolve.solve_fixed_grid(vf, u0, ts=ts, parameters=p, solver=solver)
    assert jnp.allclose(solution.t[-1], ts_reference[-1])
    assert jnp.allclose(solution.u[-1], ys_reference[-1], atol=1e-3, rtol=1e-3)
