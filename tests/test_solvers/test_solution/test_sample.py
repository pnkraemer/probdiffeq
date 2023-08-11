"""Tests for sampling behaviour."""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import smoothers
from tests.setup import setup


@testing.fixture(name="approximation")
def fixture_approximation():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = smoothers.fixedpoint_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    output_scale = jnp.ones_like(impl.ssm_util.prototype_output_scale())
    return ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
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
    # todo: remove "u" from this output?
    u, samples = approximation.posterior.sample(key, shape=shape)
    assert u.shape == shape + approximation.u.shape
    assert samples.shape == shape + impl.random.sample_shape(approximation.marginals)

    # Todo: test values of the samples by checking a chi2 statistic
    #  in terms of the joint posterior. But this requires a joint_posterior()
    #  method, which is only future work I guess. So far we use the eye-test
    #  in the notebooks, which looks good.
