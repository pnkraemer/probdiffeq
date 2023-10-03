"""Compare solve_fixed_grid to solve_and_save_every_step."""


import jax.numpy as jnp

from probdiffeq import adaptive, controls, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


def test_fixed_grid_result_matches_adaptive_grid_result():
    vf, u0, (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)
    control = controls.integral_clipped()  # Any clipped controller will do.
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2, control=control)

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale=output_scale)
    args = (vf, init)

    adaptive_kwargs = {
        "t0": t0,
        "t1": t1,
        "dt0": 0.1,
        "adaptive_solver": adaptive_solver,
    }
    solution_adaptive = ivpsolve.solve_and_save_every_step(*args, **adaptive_kwargs)

    grid_adaptive = solution_adaptive.t
    fixed_kwargs = {"grid": grid_adaptive, "solver": solver}
    solution_fixed = ivpsolve.solve_fixed_grid(*args, **fixed_kwargs)
    assert testing.tree_all_allclose(solution_adaptive, solution_fixed)
