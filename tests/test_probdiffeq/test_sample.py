"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import functools, ode, random, testing, tree_util
from probdiffeq.backend import numpy as np


@testing.fixture(name="approximation_and_strategy")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_approximation_and_strategy(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    errorest = probdiffeq.errorest_local_residual(prior=ibm, correction=ts0, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    sol = functools.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)
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
