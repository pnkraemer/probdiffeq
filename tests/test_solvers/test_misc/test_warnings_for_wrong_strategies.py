"""Some strategies don't work with all solution routines."""
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.strategies import fixedpoint, smoothers
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


def test_warning_for_fixedpoint_in_save_every_step_mode():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = fixedpoint.fixedpoint_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)

    with testing.warns():
        _ = ivpsolve.solve_and_save_every_step(
            vf, init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1
        )


def test_warning_for_smoother_in_save_at_mode():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = smoothers.smoother_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)

    with testing.warns():
        _ = ivpsolve.solve_and_save_at(
            vf,
            init,
            save_at=jnp.linspace(t0, t1),
            adaptive_solver=adaptive_solver,
            dt0=0.1,
        )
