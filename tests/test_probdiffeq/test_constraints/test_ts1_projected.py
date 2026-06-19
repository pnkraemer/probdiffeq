"""Test equivalence of different linearisation modules."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import np, ode, random, testing


@testing.parametrize("seed", [1, 2, 3])
@testing.parametrize("num_probes", [10])
def test_accuracy_matches_dense_ts1(seed, num_probes):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    strategy = probdiffeq.strategy_filter()
    save_at = np.linspace(t0, t1, num=100, endpoint=True)

    # Build the rest of the solver (projected, medium precision)
    key = random.prng_key(seed=seed)
    ssm = probdiffeq.state_space_model_blockdiag()
    prior = ssm.prior_wiener_integrated(tcoeffs)
    ode_ts1_projected = ssm.constraint_ode_ts1_projected(
        vf, key=key, num_probes=num_probes
    )
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


@testing.parametrize("seed", [1, 2])
@testing.parametrize("num_probes", [10_000])
def test_both_projected_constraints_are_identical(seed, num_probes):
    """Assert that residual-based constraints and corresponding TS1 versions match."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    strategy = probdiffeq.strategy_filter()
    grid = np.linspace(t0, t1, num=10, endpoint=True)

    # Build the rest of the solver (projected, medium precision)
    key = random.prng_key(seed=seed)
    ssm = probdiffeq.state_space_model_blockdiag_matfree(key=key, num_probes=num_probes)
    prior = ssm.prior_wiener_integrated(tcoeffs)
    ode_ts1 = ssm.constraint_ode_ts1(vf)
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_projected = solve(prior, grid=grid)

    # Build the rest of the solver (dense reference, high precision)
    ssm_dense = probdiffeq.state_space_model_dense()
    prior = ssm_dense.prior_wiener_integrated(tcoeffs)
    ode_ts1_reference = ssm_dense.constraint_ode_ts1_projected(
        vf, key=key, num_probes=num_probes
    )
    solver = probdiffeq.solver(strategy=strategy, constraint=ode_ts1_reference)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_reference = solve(prior, grid=grid)

    # Assert similarity:
    # compare highest-index taylor coeffs because these are the most sensitive
    # use a strict tol because their behaviours should match well
    expected = solution_reference.u.mean[-1]["U"].prey
    received = solution_projected.u.mean[-1]["U"].prey
    assert testing.allclose(received, expected)
