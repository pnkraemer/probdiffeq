"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, testing


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_solution_is_accurate(fact):

    @probdiffeq.ode_second_order
    def vf(u, du, /, *, t):
        del t
        del du
        return -u

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, (1.0, 0.0), t=0.0)

    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_state_std(constraint=ts0, prior=iwp, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)

    save_at = np.linspace(0.0, 5.0, endpoint=True, num=10)
    solution_2nd = func.jit(solve)(init, save_at=save_at, atol=1e-6, rtol=1e-6)

    assert testing.allclose(solution_2nd.u.mean[0], np.cos(solution_2nd.t))
