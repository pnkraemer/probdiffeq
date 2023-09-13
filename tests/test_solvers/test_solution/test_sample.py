"""Tests for sampling behaviour."""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import markov, uncalibrated
from probdiffeq.solvers.strategies import correction, fixedpoint, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="approximation")
def fixture_approximation():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = correction.ts0()
    strategy = fixedpoint.fixedpoint_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=2)
    return ivpsolve.solve_and_save_every_step(
        vf,
        tcoeffs,
        t0=t0,
        t1=t1,
        solver=solver,
        atol=1e-2,
        rtol=1e-2,
        dt0=0.1,
        output_scale=output_scale,
    )


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(approximation, shape):
    key = jax.random.PRNGKey(seed=15)
    posterior = markov.select_terminal(approximation.posterior)
    u, samples = markov.sample(key, posterior, shape=shape, reverse=True)
    assert u.shape == shape + approximation.u.shape
    assert samples.shape == shape + impl.stats.sample_shape(approximation.marginals)
