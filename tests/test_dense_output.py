"""Tests for IVP solvers."""
import jax.numpy as jnp
import jax.random
from pytest_cases import parametrize, parametrize_with_cases

from odefilter import controls, dense_output, ivpsolve


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver",
    cases=".solver_cases",
    prefix="solver_",
    has_tag=("solve", "filter"),
)
def test_offgrid_marginals_filter(vf, u0, t0, t1, p, solver):
    solution = ivpsolve.solve(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=solver, atol=1e-1, rtol=1e-1
    )

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    u_left, _ = dense_output.offgrid_marginals(
        t=solution[0].t + 1e-4,
        solution=solution[1],
        solution_previous=solution[0],
        solver=solver,
    )
    u_right, _ = dense_output.offgrid_marginals(
        t=solution[1].t - 1e-4,
        solution=solution[1],
        solution_previous=solution[0],
        solver=solver,
    )
    assert jnp.allclose(u_left, solution[0].u, atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(u_right, solution[0].u, atol=1e-3, rtol=1e-3)

    # Repeat the same but interpolating via *_searchsorted:
    # check we correctly landed in the first interval
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    u, _ = dense_output.offgrid_marginals_searchsorted(
        ts=ts, solution=solution, solver=solver
    )
    assert jnp.allclose(u[0], solution.u[0], atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(u[0], solution.u[1], atol=1e-3, rtol=1e-3)


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver",
    cases=".solver_cases",
    prefix="solver_",
    has_tag=("solve", "smoother"),
)
def test_offgrid_marginals_smoother(vf, u0, t0, t1, p, solver):

    solution = ivpsolve.solve(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
        atol=1e-1,
        rtol=1e-1,
        control=controls.ClippedIntegral(),
    )

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    u_left, _ = dense_output.offgrid_marginals(
        t=solution[0].t + 1e-4,
        solution=solution[1],
        solution_previous=solution[0],
        solver=solver,
    )
    u_right, _ = dense_output.offgrid_marginals(
        t=solution[1].t - 1e-4,
        solution=solution[1],
        solution_previous=solution[0],
        solver=solver,
    )
    assert jnp.allclose(u_left, solution[0].u, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(u_right, solution[1].u, atol=1e-3, rtol=1e-3)

    # Repeat the same but interpolating via *_searchsorted:
    # check we correctly landed in the first interval
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    u, _ = dense_output.offgrid_marginals_searchsorted(
        ts=ts, solution=solution, solver=solver
    )
    assert jnp.allclose(u[0], solution.u[0], atol=1e-3, rtol=1e-3)
    assert jnp.allclose(u[-1], solution.u[-1], atol=1e-3, rtol=1e-3)


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver",
    cases=".solver_cases",
    prefix="solver_",
    has_tag=["checkpoint", "smoother"],
)
@parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_grid_samples(vf, u0, t0, t1, p, solver, shape):
    ts = jnp.linspace(t0, t1, num=20, endpoint=True)
    solution = ivpsolve.simulate_checkpoints(
        vf, u0, ts=ts, parameters=p, solver=solver, atol=1e-1, rtol=1e-1
    )
    key = jax.random.PRNGKey(seed=15)

    u, samples = dense_output.sample(key, solution=solution, solver=solver, shape=shape)
    assert u.shape == shape + solution.u.shape
    assert samples.shape == shape + solution.marginals.mean.shape

    # Todo: test values of the samples by checking a chi2 statistic
    #  in terms of the joint posterior. But this requires a joint_posterior()
    #  method, which is only future work I guess. So far we use the eye-test
    #  in the notebooks, which looks good.


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver", cases=".solver_cases", prefix="solver_", has_tag=("checkpoint", "dense")
)
def test_negative_marginal_log_likelihood(vf, u0, t0, t1, p, solver):
    ts = jnp.linspace(t0, t1, num=5)
    solution = ivpsolve.simulate_checkpoints(vf, u0, ts=ts, parameters=p, solver=solver)

    data = solution.u + 0.005
    k, d = solution.u.shape

    mll = dense_output.negative_marginal_log_likelihood(
        observation_std=jnp.ones((k, d)), u=data, solution=solution, solver=solver
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)
