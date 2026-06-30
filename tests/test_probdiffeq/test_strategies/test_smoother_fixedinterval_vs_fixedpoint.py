"""The fixedpoint-smoother and smoother should yield identical results.

That is, when called with correct adaptive- and checkpoint-setups.
"""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, ode, testing
from probdiffeq.util import test_util


@testing.fixture(name="solver_setup")
@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_blockdiag,
    ],
)
def fixture_solver_setup(ssm_factory):
    """Set up the Lotka-Volterra IVP and jet expansion for the smoother equivalence tests."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    return {
        "vf": vf,
        "tcoeffs": tcoeffs,
        "t0": t0,
        "t1": t1,
        "ssm_factory": ssm_factory,
    }


@testing.fixture(name="solution_smoother")
def fixture_solution_smoother(solver_setup):
    """Solve adaptively with the fixed interval smoother using save-every-step."""
    tcoeffs = solver_setup["tcoeffs"]
    ssm = solver_setup["ssm_factory"]()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(solver_setup["vf"])
    strategy = probdiffeq.strategy_smoother_fixedinterval()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)
    solve = test_util.solve_adaptive_save_every_step(error=error, solver=solver)
    t0, t1 = solver_setup["t0"], solver_setup["t1"]
    return solve(iwp, t0=t0, t1=t1, dt0=0.1, atol=1e-3, rtol=1e-3)


def test_fixedpoint_smoother_equivalent_same_grid(
    solver_setup, solution_smoother
) -> None:
    """Test that with save_at=smoother_solution.t, the results should be identical."""
    tcoeffs = solver_setup["tcoeffs"]
    ssm = solver_setup["ssm_factory"]()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(solver_setup["vf"])
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)

    save_at = solution_smoother.t
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    solution_fixedpoint = func.jit(solve)(
        iwp, save_at=save_at, dt0=0.1, atol=1e-3, rtol=1e-3
    )

    sol_fp, sol_sm = solution_fixedpoint, solution_smoother  # alias for brevity
    assert testing.allclose(sol_fp.t, sol_sm.t)
    assert testing.allclose(sol_fp.u.mean, sol_sm.u.mean)
    assert testing.allclose(sol_fp.u.std, sol_sm.u.std)
    assert testing.allclose(sol_fp.u, sol_sm.u)
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
    cond_fp = func.vmap(lambda c: c.preconditioner_apply())(cond_fp)
    cond_sm = func.vmap(lambda c: c.preconditioner_apply())(cond_sm)
    assert testing.allclose(cond_fp, cond_sm)
