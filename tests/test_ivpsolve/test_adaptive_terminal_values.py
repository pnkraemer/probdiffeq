"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, testing, tree_util


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_output_matches_reference(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Don't try all solvers because they're tested in a different file.
    # This test here is only to assert that terminal-value simulation works.
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    constraint = probdiffeq.constraint_ode_ts0(ssm=ssm)
    solver = probdiffeq.solver_dynamic(
        vf, strategy=strategy, prior=iwp, constraint=constraint, ssm=ssm
    )
    errorest = probdiffeq.errorest_local_residual_cached(prior=iwp, ssm=ssm)

    # Compute the PN solution
    dt0 = ivpsolve.dt0_adaptive(
        vf, u0, t0, error_contraction_rate=5, rtol=1e-3, atol=1e-3
    )

    solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, errorest=errorest)
    received = func.jit(solve)(init, t0=t0, t1=t1, dt0=dt0, atol=1e-3, rtol=1e-3)

    # Compute a reference solution
    save_at = np.asarray([t0, t1])
    expected = ode.odeint_and_save_at(vf, u0, save_at=save_at, atol=1e-7, rtol=1e-7)
    expected = tree_util.tree_map(lambda s: s[-1], expected)

    # The results should be very similar
    assert testing.allclose(received.u.mean[0], expected)

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, received.u.mean)
    u_std_shape = tree_util.tree_map(np.shape, received.u.std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)
