"""Compare simulate_terminal_values to solve_adaptive_save_every_step."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def test_terminal_values_identical(fact):
    """The terminal values must be identical."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
    asolver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
    solution_loop = ivpsolve.solve_adaptive_save_every_step(
        init, t0=t0, t1=t1, adaptive_solver=asolver, dt0=0.1, ssm=ssm
    )
    expected = tree_util.tree_map(lambda s: s[-1], solution_loop)

    received = ivpsolve.solve_adaptive_terminal_values(
        init, t0=t0, t1=t1, adaptive_solver=asolver, dt0=0.1, ssm=ssm
    )
    assert testing.allclose(received, expected)

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, received.u)
    u_std_shape = tree_util.tree_map(np.shape, received.u_std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)
