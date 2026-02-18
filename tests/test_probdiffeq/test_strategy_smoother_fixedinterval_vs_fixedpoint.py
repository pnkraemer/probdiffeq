"""The fixedpoint-smoother and smoother should yield identical results.

That is, when called with correct adaptive- and checkpoint-setups.
"""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, testing, tree
from probdiffeq.util import test_util


@testing.fixture(name="solver_setup")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solver_setup(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    return {"vf": vf, "tcoeffs": tcoeffs, "t0": t0, "t1": t1, "fact": fact}


@testing.fixture(name="solution_smoother")
def fixture_solution_smoother(solver_setup):
    tcoeffs, fact = solver_setup["tcoeffs"], solver_setup["fact"]
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(solver_setup["vf"], ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual(prior=ibm, ssm=ssm)
    solve = test_util.solve_adaptive_save_every_step(error=error, solver=solver)
    t0, t1 = solver_setup["t0"], solver_setup["t1"]
    return solve(init, t0=t0, t1=t1, dt0=0.1, atol=1e-3, rtol=1e-3)


def test_fixedpoint_smoother_equivalent_same_grid(solver_setup, solution_smoother):
    """Test that with save_at=smoother_solution.t, the results should be identical."""
    tcoeffs, fact = solver_setup["tcoeffs"], solver_setup["fact"]
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(solver_setup["vf"], ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual(prior=ibm, ssm=ssm)

    save_at = solution_smoother.t
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    solution_fixedpoint = func.jit(solve)(
        init, save_at=save_at, dt0=0.1, atol=1e-3, rtol=1e-3
    )

    sol_fp, sol_sm = solution_fixedpoint, solution_smoother  # alias for brevity
    assert testing.allclose(sol_fp.t, sol_sm.t)
    assert testing.allclose(sol_fp.u.mean, sol_sm.u.mean)
    assert testing.allclose(sol_fp.u.std, sol_sm.u.std)
    assert testing.allclose(sol_fp.u.marginals, sol_sm.u.marginals)
    assert testing.allclose(sol_fp.num_steps, sol_sm.num_steps)
    assert testing.allclose(
        sol_fp.solution_full.marginal, sol_sm.solution_full.marginal
    )

    # The backward conditionals use different parametrisations
    # but implement the same transitions
    cond_fp, cond_sm = (
        sol_fp.solution_full.conditional,
        sol_sm.solution_full.conditional,
    )
    cond_fp = func.vmap(ssm.conditional.preconditioner_apply)(cond_fp)
    cond_sm = func.vmap(ssm.conditional.preconditioner_apply)(cond_sm)
    assert testing.allclose(cond_fp, cond_sm)


def test_fixedpoint_smoother_equivalent_different_grid(solver_setup, solution_smoother):
    """Test that the interpolated smoother result equals the save_at result."""
    save_at = solution_smoother.t

    # Re-generate the smoothing solver
    tcoeffs, fact = solver_setup["tcoeffs"], solver_setup["fact"]
    _init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(solver_setup["vf"], ssm=ssm)
    strategy_sm = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
    solver_smoother = probdiffeq.solver(
        strategy=strategy_sm, prior=ibm, constraint=ts0, ssm=ssm
    )

    # Compute the offgrid-marginals
    ts = np.linspace(save_at[0], save_at[-1], num=7, endpoint=True)
    offgrid = func.partial(
        solver_smoother.offgrid_marginals, solution=solution_smoother
    )
    interpolated = func.jit(func.vmap(offgrid))(ts[1:-1])

    # Generate a fixedpoint solver and solve (saving at the interpolation points)
    tcoeffs, fact = solver_setup["tcoeffs"], solver_setup["fact"]
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(solver_setup["vf"], ssm=ssm)
    strategy_fp = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy_fp, prior=ibm, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual(prior=ibm, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    solution_fixedpoint = func.jit(solve)(
        init, save_at=ts, dt0=0.1, atol=1e-3, rtol=1e-3
    )

    # Extract the interior points of the save_at solution
    # (because only there is the interpolated solution defined)
    u_fixedpoint = tree.tree_map(lambda s: s[1:-1], solution_fixedpoint.u.mean)
    u_std_fixedpoint = tree.tree_map(lambda s: s[1:-1], solution_fixedpoint.u.std)
    marginals_fixedpoint = tree.tree_map(
        lambda s: s[1:-1], solution_fixedpoint.u.marginals
    )

    assert testing.allclose(u_fixedpoint, interpolated.mean)
    assert testing.allclose(u_std_fixedpoint, interpolated.std)

    # Compare QOI and marginals
    marginals_allclose_func = func.partial(testing.marginals_allclose, ssm=ssm)
    marginals_allclose_func = func.vmap(marginals_allclose_func)
    assert np.all(marginals_allclose_func(marginals_fixedpoint, interpolated.marginals))
