"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest
from jax.experimental.ode import odeint
from pytest_cases import parametrize_with_cases

from odefilter import ivpsolve, solvers
from odefilter.strategies import smoothers


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver", cases=".solver_cases", prefix="solver_", has_tag=("checkpoint",)
)
def test_simulate_checkpoints(vf, u0, t0, t1, p, solver):
    ts = jnp.linspace(t0, t1, num=10)

    odeint_solution = odeint(
        lambda y, t, *par: vf(y, t=t, p=par), u0[0], ts, *p, atol=1e-6, rtol=1e-6
    )
    ts_reference, ys_reference = ts, odeint_solution
    solution = ivpsolve.simulate_checkpoints(
        vf, u0, ts=ts, parameters=p, solver=solver, atol=1e-4, rtol=1e-4
    )
    assert jnp.allclose(solution.t, ts)
    assert jnp.allclose(solution.t, ts_reference)
    assert jnp.allclose(solution.u, ys_reference, atol=1e-3, rtol=1e-3)


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
def test_smoother_warning(vf, u0, t0, t1, p):
    """A non-fixed-point smoother is not usable in checkpoint-simulation."""
    ts = jnp.linspace(t0, t1, num=3)
    solver = solvers.DynamicSolver(strategy=smoothers.Smoother())

    with pytest.warns():
        ivpsolve.simulate_checkpoints(vf, u0, ts=ts, parameters=p, solver=solver)
