"""Compare simulate_terminal_values to solve_adaptive_save_every_step."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import ode, testing, tree_util


@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def test_terminal_values_identical(fact):
    """The terminal values must be identical."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ibm, ts0, ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, ssm=ssm)
    asolver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

    init = solver.initial_condition()

    args = (vf, init)
    kwargs = {"t0": t0, "t1": t1, "adaptive_solver": asolver, "dt0": 0.1, "ssm": ssm}

    solution_loop = ivpsolve.solve_adaptive_save_every_step(*args, **kwargs)
    expected = tree_util.tree_map(lambda s: s[-1], solution_loop)

    received = ivpsolve.solve_adaptive_terminal_values(*args, **kwargs)
    assert testing.tree_all_allclose(received, expected)
