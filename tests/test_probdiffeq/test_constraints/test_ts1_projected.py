"""Test equivalence of different linearisation modules."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import np, ode, testing


def test_residual_matches_ts1():
    """Assert that residual-based constraints and corresponding TS1 versions match."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    strategy = probdiffeq.strategy_filter()
    save_at = np.linspace(t0, t1, num=100, endpoint=True)

    # Build the rest of the solver (projected, medium precision)
    ssm = probdiffeq.state_space_model_blockdiag()
    prior = ssm.prior_wiener_integrated(tcoeffs)
    ode_ts1_projected = ssm.constraint_ode_ts1_projected(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1_projected)
    error = probdiffeq.error_state_std(constraint=ode_ts1_projected)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution_projected = solve(prior, save_at=save_at, atol=1e-4, rtol=1e-4)

    # Build the rest of the solver (dense reference, high precision)
    ssm_dense = probdiffeq.state_space_model_dense()
    prior = ssm_dense.prior_wiener_integrated(tcoeffs)
    ode_ts1_reference = ssm_dense.constraint_ode_ts1(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1_reference)
    error = probdiffeq.error_state_std(constraint=ode_ts1_reference)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution_reference = solve(prior, save_at=save_at, atol=1e-8, rtol=1e-8)

    # Assert similarity (tol slightly weaker than simulation tol)
    expected = solution_reference.u.mean[0]
    received = solution_projected.u.mean[0]
    assert testing.allclose(received, expected, atol=1e-3, rtol=1e-3)
