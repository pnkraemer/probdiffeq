"""Test equivalence of different linearisation modules."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import np, ode


def test_residual_matches_ts1():
    """Assert that residual-based constraints and corresponding TS1 versions match."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    ssm = probdiffeq.state_space_model_dense()
    vf = probdiffeq.ode(vf)
    ode_ts1 = ssm.constraint_ode_ts1_projected(vf)

    # Build the rest of the solver
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1)
    error = probdiffeq.error_state_std(constraint=ode_ts1)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)

    # Solve the ODE. Try different solution routines.
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    prior = ssm.prior_wiener_integrated(jetexpand(vf, [u0], t=t0)[0])
    save_at = np.linspace(t0, t1, num=100, endpoint=True)
    solution = solve(prior, save_at=save_at, atol=1e-4, rtol=1e-2)
    print(solution.u.mean[0])

    assert False
