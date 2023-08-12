"""Compare solve_fixed_grid to solve_and_save_every_step."""


import jax.numpy as jnp

from probdiffeq import controls, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


def test_fixed_grid_result_matches_adaptive_grid_result():
    vf, u0, (t0, t1) = setup.ode()

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=2)
    args = (vf, tcoeffs)

    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)
    adaptive_kwargs = {
        "t0": t0,
        "t1": t1,
        "dt0": 0.1,
        "solver": solver,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        # Any clipped controller will do.
        "control": controls.integral_clipped(),
    }
    solution_adaptive = ivpsolve.solve_and_save_every_step(*args, **adaptive_kwargs)

    grid_adaptive = solution_adaptive.t
    fixed_kwargs = {
        "grid": grid_adaptive,
        "solver": solver,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
    }
    solution_fixed = ivpsolve.solve_fixed_grid(*args, **fixed_kwargs)
    assert testing.tree_all_allclose(solution_adaptive, solution_fixed)