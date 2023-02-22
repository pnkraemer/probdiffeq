"""Interact with estimated initial value problem (IVP) solutions on dense grids."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


def sample(key, *, solution, solver, shape=()):
    return solver.strategy.sample(key, posterior=solution.posterior, shape=shape)


# todo: the functions herein should only depend on posteriors / strategies!


def offgrid_marginals_searchsorted(*, ts, solution, solver):
    """Compute off-grid marginals on a dense grid via jax.numpy.searchsorted.

    !!! warning
        The elements in ts and the elements in the solution grid must be disjoint.
        Otherwise, anything can happen and the solution will be incorrect.
        At the moment, we do not check this.

    !!! warning
        The elements in ts must be strictly in (t0, t1).
        They must not lie outside the interval, and they must not coincide
        with the interval boundaries.
        At the moment, we do not check this.
    """
    # todo: support "method" argument to be passed to searchsorted.

    # side="left" and side="right" are equivalent
    # because we _assume_ that the point sets are disjoint.
    indices = jnp.searchsorted(solution.t, ts)

    # Solution slicing to the rescue
    solution_left = solution[indices - 1]
    solution_right = solution[indices]

    # Vmap to the rescue :) It does not like kw-only arguments, though.
    @jax.vmap
    def marginals_vmap(sprev, t, s):
        return offgrid_marginals(
            t=t, solution=s, solution_previous=sprev, solver=solver
        )

    return marginals_vmap(solution_left, ts, solution_right)


def offgrid_marginals(*, solution, t, solution_previous, solver):
    return solver.strategy.offgrid_marginals(
        marginals=solution.marginals,
        posterior_previous=solution_previous.posterior,
        t=t,
        t0=solution_previous.t,
        t1=solution.t,
        scale_sqrtm=solution.output_scale_sqrtm,
    )


class _NMLLState(NamedTuple):
    rv: Any
    num_data: int
    nmll: float


def negative_marginal_log_likelihood(*, observation_std, u, solution):
    """Compute the negative marginal log-likelihood of \
     observations of the IVP solution."""
    # todo: complain if it is used with a filter, not a smoother?

    bw_models = jax.tree_util.tree_map(
        lambda x: x[1:, ...], solution.posterior.backward_model
    )
    init = jax.tree_util.tree_map(lambda x: x[-1, ...], solution.posterior.init)

    def filter_step(carry, x):
        # Read
        rv, num_data, nmll_prev = carry.rv, carry.num_data, carry.nmll
        bw_model, obs_std, data = x

        # Extrapolate
        rv_ext = bw_model.marginalise(rv)

        # Correct (with an alias for long function names)
        obs, (cor, _) = rv_ext.condition_on_qoi_observation(
            data, observation_std=obs_std
        )

        # Compute marginal log likelihood (with an alias for long function names)
        nmll_new = -1 * jnp.sum(obs.logpdf(data))
        nmll_updated = (num_data * nmll_prev + nmll_new) / (num_data + 1)

        # Return values
        x = _NMLLState(cor, num_data + 1, nmll_updated)
        return x, x

    # todo: this should return a Filtering posterior or a smoothing posterior
    #  which could then be plotted. Right?
    #  (We might also want some dense-output/checkpoint kind of thing here)
    # todo: we should reuse the extrapolation model implementations.
    #  But this only works if the ODE posterior uses the preconditioner (I think).
    # todo: we should allow proper noise, and proper information functions.
    #  But it is not clear which data structure that should be.
    #
    (_, _, nmll), _ = jax.lax.scan(
        f=filter_step,
        init=_NMLLState(init, 0, 0.0),
        xs=(bw_models, observation_std[:-1], u[:-1]),  # todo: use all data!
        reverse=True,
    )
    return nmll
