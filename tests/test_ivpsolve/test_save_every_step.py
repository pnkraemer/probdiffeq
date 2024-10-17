"""Assert that solve_with_python_loop is accurate."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing
from probdiffeq.impl import impl


@testing.fixture(name="python_loop_solution")
def fixture_python_loop_solution(ssm):
    vf, u0, (t0, t1) = ssm.default_ode

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale=output_scale)

    ts0 = ivpsolvers.correction_ts0()
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_mle(strategy)
    control = ivpsolve.control_proportional_integral(clip=True)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, control=control)

    dt0 = ivpsolve.dt0_adaptive(
        vf, u0, t0=t0, atol=1e-2, rtol=1e-2, error_contraction_rate=5
    )

    init = solver.initial_condition()

    args = (vf, init)
    kwargs = {"t0": t0, "t1": t1, "adaptive_solver": adaptive_solver, "dt0": dt0}
    return ivpsolve.solve_adaptive_save_every_step(*args, **kwargs)


@testing.fixture(name="reference_solution")
def fixture_reference_solution(ssm):
    vf, u0, (t0, t1) = ssm.default_ode
    return ode.odeint_dense(vf, u0, t0=t0, t1=t1, atol=1e-10, rtol=1e-10)


def test_python_loop_output_matches_reference(python_loop_solution, reference_solution):
    expected = reference_solution(python_loop_solution.t)
    received = python_loop_solution.u
    assert np.allclose(received, expected, rtol=1e-2)
