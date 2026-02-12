"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import ode, random, testing, tree_util


@testing.fixture(name="approximation_and_strategy")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_approximation_and_strategy(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.correction_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother(ssm=ssm)
    solver = probdiffeq.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    errorest = probdiffeq.errorest_schober_bosch(
        prior=ibm, correction=ts0, atol=1e-2, rtol=1e-2, ssm=ssm
    )
    sol = ivpsolve.solve_adaptive_save_every_step(
        init, t0=t0, t1=t1, solver=solver, errorest=errorest
    )
    return sol, strategy


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(approximation_and_strategy, shape):
    approximation, strategy = approximation_and_strategy

    key = random.prng_key(seed=15)
    samples = strategy.markov_sample(
        key, approximation.posterior, shape=shape, reverse=True
    )
    for s, u in zip(samples, approximation.u.mean):
        s_shape = tree_util.tree_map(lambda x: x.shape, s)
        u_inner_shape = tree_util.tree_map(lambda x: shape + x.shape, u)
        assert testing.allclose(s_shape, u_inner_shape)
