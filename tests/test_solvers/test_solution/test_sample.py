"""Tests for sampling behaviour."""
import jax
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import markov, uncalibrated
from probdiffeq.solvers.strategies import smoothers
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="approximation")
def fixture_approximation():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = smoothers.smoother_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.solve_and_save_every_step(
        vf, init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1
    )


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(approximation, shape):
    key = jax.random.PRNGKey(seed=15)
    posterior = markov.select_terminal(approximation.posterior)
    (u, samples), (u_init, samples_init) = markov.sample(
        key, posterior, shape=shape, reverse=True
    )
    margs = jax.tree_util.tree_map(lambda x: x[1:], approximation.marginals)
    assert u.shape == shape + approximation.u[1:].shape
    assert samples.shape == shape + impl.stats.sample_shape(margs)

    margs = jax.tree_util.tree_map(lambda x: x[0], approximation.marginals)
    assert u_init.shape == shape + approximation.u[0].shape
    assert samples_init.shape == shape + impl.stats.sample_shape(margs)
