"""Tests for sampling behaviour."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import smoothers


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.case()
def case_isotropic_factorisation():
    def iso_factory(ode_shape, num_derivatives):
        return recipes.ts0_iso(num_derivatives=num_derivatives)

    return iso_factory, 2.0


@testing.case()  # this implies success of the scalar solver
def case_blockdiag_factorisation():
    return recipes.ts0_blockdiag, jnp.ones((2,)) * 2.0


@testing.case()
def case_dense_factorisation():
    return recipes.ts0_dense, 2.0


@testing.fixture(name="approximate_solution")
@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
def fixture_approximate_solution(problem, factorisation):
    vf, u0, (t0, t1), f_args = problem
    impl_factory, output_scale = factorisation
    solver = test_util.generate_solver(
        num_derivatives=1,
        impl_factory=impl_factory,
        strategy_factory=smoothers.smoother,
        ode_shape=jnp.shape(u0),
    )
    sol = ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        output_scale=output_scale,
        atol=1e-2,
        rtol=1e-2,
    )
    return sol, solver


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(approximate_solution, shape):
    sol, solver = approximate_solution

    key = jax.random.PRNGKey(seed=15)
    # todo: remove "u" from this output?
    u, samples = sol.posterior.sample(key, shape=shape)
    assert u.shape == shape + sol.u.shape
    assert samples.shape == shape + sol.marginals.sample_shape

    # Todo: test values of the samples by checking a chi2 statistic
    #  in terms of the joint posterior. But this requires a joint_posterior()
    #  method, which is only future work I guess. So far we use the eye-test
    #  in the notebooks, which looks good.
