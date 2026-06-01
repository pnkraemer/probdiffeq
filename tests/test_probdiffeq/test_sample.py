"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, random, testing, tree


@testing.fixture(name="solution_and_ssm")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution_and_ssm(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = probdiffeq.jetexpand_ode_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)

    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    save_at = np.linspace(t0, t1, endpoint=True, num=12)
    solution = func.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)
    return solution, ssm


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(solution_and_ssm, shape) -> None:
    solution, ssm = solution_and_ssm
    key = random.prng_key(seed=15)
    samples = solution.solution_full.sample(key, ssm=ssm, shape=shape)

    for s, u in zip(samples, solution.u.mean):
        s_shape = tree.tree_map(lambda x: x.shape, s)
        u_inner_shape = tree.tree_map(lambda x: shape + x.shape, u)
        assert testing.allclose(s_shape, u_inner_shape)
