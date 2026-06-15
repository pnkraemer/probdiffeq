"""Test equivalence of different linearisation modules."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import np, ode, testing


def test_residual_matches_ts1():
    """Assert that residual-based constraints and corresponding TS1 versions match."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    ssm = probdiffeq.state_space_model_dense()
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    prior = ssm.prior_wiener_integrated(jetexpand(vf, [u0], t=t0)[0])
    strategy = probdiffeq.strategy_filter()
    save_at = np.linspace(t0, t1, num=100, endpoint=True)

    # Build the rest of the solver
    ode_ts1_projected = ssm.constraint_ode_ts1_projected(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1_projected)
    error = probdiffeq.error_state_std(constraint=ode_ts1_projected)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution_projected = solve(prior, save_at=save_at, atol=1e-4, rtol=1e-4)

    ode_ts1_reference = ssm.constraint_ode_ts1(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1_reference)
    error = probdiffeq.error_state_std(constraint=ode_ts1_reference)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution_reference = solve(prior, save_at=save_at, atol=1e-8, rtol=1e-8)

    assert testing.allclose(
        solution_projected.u.mean[0], solution_reference.u.mean[0], atol=1e-3, rtol=1e-3
    )
