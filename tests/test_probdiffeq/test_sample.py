"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, random, testing, tree


@testing.fixture(name="approximation_and_strategy")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_approximation_and_strategy(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm)

    error = probdiffeq.error_residual(prior=ibm, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    sol = func.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)
    return sol, strategy


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(approximation_and_strategy, shape):
    approximation, strategy = approximation_and_strategy

    key = random.prng_key(seed=15)
    samples = strategy.markov_sample(
        key, approximation.solution_full, shape=shape, reverse=True
    )
    for s, u in zip(samples, approximation.u.mean):
        s_shape = tree.tree_map(lambda x: x.shape, s)
        u_inner_shape = tree.tree_map(lambda x: shape + x.shape, u)
        assert testing.allclose(s_shape, u_inner_shape)
