"""Some strategies don't work with all solution routines."""

from probdiffeq import ivpsolve, solvers
from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.taylor import autodiff
from tests.setup import setup


def test_warning_for_fixedpoint_in_save_every_step_mode():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = solvers.prior_ibm(num_derivatives=2)
    ts0 = solvers.correction_ts0()
    strategy = solvers.strategy_fixedpoint(ibm, ts0)
    solver = solvers.solver(strategy)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)

    with testing.warns():
        _ = ivpsolve.solve_adaptive_save_every_step(
            vf, init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1
        )


def test_warning_for_smoother_in_save_at_mode():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = solvers.prior_ibm(num_derivatives=2)
    ts0 = solvers.correction_ts0()
    strategy = solvers.strategy_smoother(ibm, ts0)
    solver = solvers.solver(strategy)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)

    with testing.warns():
        _ = ivpsolve.solve_adaptive_save_at(
            vf,
            init,
            save_at=np.linspace(t0, t1),
            adaptive_solver=adaptive_solver,
            dt0=0.1,
        )
