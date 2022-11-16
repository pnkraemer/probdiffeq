"""Tests for solving IVPs for checkpoints."""

import jax.numpy as jnp
import pytest

from odefilter import ivpsolve, solvers
from odefilter.implementations import recipes
from odefilter.strategies import smoothers


def test_checkpoints_simulated_correctly(
    reference_checkpoints, solution_checkpoints, tolerances
):
    t_ref, u_ref = reference_checkpoints
    atol, rtol = tolerances
    solution, _ = solution_checkpoints

    assert jnp.allclose(solution.t, t_ref)
    assert jnp.allclose(solution.u, u_ref, atol=atol, rtol=rtol)


def test_smoother_warning(ode_problem):
    """A non-fixed-point smoother is not usable in checkpoint-simulation."""
    vf, u0, t0, t1, p = ode_problem
    ts = jnp.linspace(t0, t1, num=3)
    solver = solvers.DynamicSolver(smoothers.Smoother(recipes.IsoTS0.from_params()))

    # todo: does this compute the full solve? We only want to catch a warning!
    with pytest.warns():
        ivpsolve.simulate_checkpoints(vf, u0, ts=ts, parameters=p, solver=solver)
