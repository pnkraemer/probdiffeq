"""Test equivalence of different linearisation modules."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import np, ode, testing


def test_accuracy_matches_dense_ts1():
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    strategy = probdiffeq.strategy_filter()
    save_at = np.linspace(t0, t1, num=100, endpoint=True)

    # Build the rest of the solver (projected, medium precision)
    ssm = probdiffeq.state_space_model_dense()
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


def test_both_projected_constraints_are_identical():
    """Assert that residual-based constraints and corresponding TS1 versions match."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    strategy = probdiffeq.strategy_filter()
    save_at = np.linspace(t0, t1, num=7, endpoint=True)

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
    ode_ts1_reference = ssm_dense.constraint_ode_ts1_projected(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1_reference)
    error = probdiffeq.error_state_std(constraint=ode_ts1_reference)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution_reference = solve(prior, save_at=save_at, atol=1e-4, rtol=1e-4)

    # Assert similarity:
    # compare highest-index taylor coeffs because these are the most sensitive
    # use a strict tol because their behaviours should match well
    expected = solution_reference.u.mean[-1]["U"].prey
    received = solution_projected.u.mean[-1]["U"].prey
    print(expected)
    print()
    print(received)
    assert testing.allclose(received, expected)
