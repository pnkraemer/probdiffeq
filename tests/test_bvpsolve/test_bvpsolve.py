"""Tests for BVP solver."""

import jax
import jax.numpy as jnp

from probdiffeq import bvpsolve
from probdiffeq.backend import statespace


def test_solve_separable_affine_2nd(num_derivatives=4):
    # Set up a prior
    grid = jnp.linspace(0.0, 1.0, endpoint=True, num=20)
    prior = extra.ibm_discretise_fwd(jnp.diff(grid), num_derivatives=num_derivatives)

    # Set up a problem
    eps = 1e-2
    g0, g1 = (1.0, -1.0), (1.0, 0.0)
    ode = (jnp.ones_like(grid) / eps, jnp.zeros_like(grid))

    # Solve the BVP
    solution = bvpsolve.solve_separable_affine_2nd(ode, bconds=(g0, g1), prior=prior)

    def true_sol(t):
        a = jnp.exp(-t / jnp.sqrt(eps))
        b = jnp.exp((t - 2.0) / jnp.sqrt(eps))
        c = jnp.exp(-2.0 / jnp.sqrt(eps))
        return (a - b) / (1 - c)

    means, stds = _marginal_moments(solution)
    assert jnp.allclose(means[:, 0], true_sol(grid[1:]), atol=1e-3)


def _marginal_moments(precon_mseq):
    def step(carry, input):
        trans, prec = input
        rv = extra.extrapolate_precon(carry, trans, prec)
        return rv, rv

    _, rvs = jax.lax.scan(
        step,
        init=precon_mseq.init,
        xs=(precon_mseq.conditional, precon_mseq.preconditioner),
        reverse=False,
    )
    means, cov_sqrtms = rvs.mean, rvs.cov_sqrtm_lower

    @jax.vmap
    def cov(x):
        return x @ x.T

    covs = cov(cov_sqrtms)
    stds = jnp.sqrt(jax.vmap(jnp.diagonal)(covs))
    return means, stds
