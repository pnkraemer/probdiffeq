"""Assert that the dynamic solver can successfully solve a linear function.

Specifically, we solve a linear function with exponentially increasing output-scale.
This is difficult for the MLE- and calibration-free solver,
but not for the dynamic solver.
"""
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from tests.setup import setup


def test_exponential_approximated_well():
    vf, u0, (t0, t1), solution = setup.ode_affine()

    ibm = priors.ibm_adaptive(num_derivatives=1)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.dynamic(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition((*u0, vf(*u0, t=t0)), output_scale=output_scale)

    problem_args = (vf, init)
    grid = jnp.linspace(t0, t1, num=20)
    solver_kwargs = {"grid": grid, "solver": solver}
    approximation = ivpsolve.solve_fixed_grid(*problem_args, **solver_kwargs)

    rmse = _rmse(approximation.u[-1], solution(t1))
    assert rmse < 0.1


def _rmse(a, b):
    return jnp.linalg.norm((a - b) / b) / jnp.sqrt(b.size)
