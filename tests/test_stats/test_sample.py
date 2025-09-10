"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import ode, random, testing, tree_util


@testing.fixture(name="approximation")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_approximation(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_smoother(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
    return ivpsolve.solve_adaptive_save_every_step(
        init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
    )


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(approximation, shape):
    key = random.prng_key(seed=15)
    posterior = stats.markov_select_terminal(approximation.posterior)
    samples, samples_init = stats.markov_sample(
        key, posterior, shape=shape, reverse=True, ssm=approximation.ssm
    )

    for i, s, u in zip(samples_init, samples, approximation.u):
        i_shape = tree_util.tree_map(lambda x: x.shape, i)
        s_shape = tree_util.tree_map(lambda x: x.shape, s)
        u_terminal_shape = tree_util.tree_map(lambda x: shape + x[-1].shape, u)
        u_inner_shape = tree_util.tree_map(lambda x: shape + x[:-1].shape, u)

        assert testing.tree_all_allclose(i_shape, u_terminal_shape)
        assert testing.tree_all_allclose(s_shape, u_inner_shape)
