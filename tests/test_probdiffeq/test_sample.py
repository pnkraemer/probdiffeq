"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, random, testing, tree


@testing.fixture(name="solution")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, (u0,), t=t0)

    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    init, iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0)

    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    save_at = np.linspace(t0, t1, endpoint=True, num=12)
    return func.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(solution, shape) -> None:
    key = random.prng_key(seed=15)
    samples = solution.solution_full.sample(key, shape=shape)

    for s, u in zip(samples, solution.u.mean):
        s_shape = tree.tree_map(lambda x: x.shape, s)
        u_inner_shape = tree.tree_map(lambda x: shape + x.shape, u)
        assert testing.allclose(s_shape, u_inner_shape)
