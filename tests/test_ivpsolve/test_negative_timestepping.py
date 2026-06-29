"""Test equivalence of different linearisation modules."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import np, ode, testing


def test_time_reversal_matches_modified_problem():
    """Test that the matfree Ts1 extension yields accurate solutions."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_unroll(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    save_at = np.linspace(t0, t1, num=20, endpoint=True)
    save_at = save_at[::-1]

    # Build the rest of the solver (dense reference, high precision)
    ssm_dense = probdiffeq.state_space_model_dense()
    prior = ssm_dense.prior_wiener_integrated(tcoeffs)
    ode_ts1_reference = ssm_dense.constraint_ode_ts1(vf)
    solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=ode_ts1_reference)
    error = probdiffeq.error_state_std(constraint=ode_ts1_reference)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    received = solve(prior, save_at=save_at, atol=1e-8, rtol=1e-8)

    # Compute a reference solution
    expected = ode.odeint_and_save_at(vf, (u0,), save_at=save_at, atol=1e-7, rtol=1e-7)

    print(received.u.mean[0]["U"].prey)

    print(expected["U"].prey)
    assert testing.allclose(received.u.mean[0], expected, atol=1e-3, rtol=1e-3)
