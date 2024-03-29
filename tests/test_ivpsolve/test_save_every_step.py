"""Assert that solve_with_python_loop is accurate."""

from probdiffeq import adaptive, ivpsolve, timestep
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="python_loop_solution")
def fixture_python_loop_solution():
    vf, u0, (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=4)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    dt0 = timestep.initial_adaptive(
        vf, u0, t0=t0, atol=1e-2, rtol=1e-2, error_contraction_rate=5
    )

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), u0, num=4)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale=output_scale)

    args = (vf, init)
    kwargs = {"t0": t0, "t1": t1, "adaptive_solver": adaptive_solver, "dt0": dt0}
    return ivpsolve.solve_and_save_every_step(*args, **kwargs)


@testing.fixture(name="reference_solution")
def fixture_reference_solution():
    vf, (u0,), (t0, t1) = setup.ode()
    return ode.odeint_dense(vf, (u0,), t0=t0, t1=t1, atol=1e-10, rtol=1e-10)


def test_python_loop_output_matches_reference(python_loop_solution, reference_solution):
    expected = reference_solution(python_loop_solution.t)
    received = python_loop_solution.u
    assert np.allclose(received, expected, rtol=1e-2)
