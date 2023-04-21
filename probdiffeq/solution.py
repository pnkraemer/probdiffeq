"""Interact with estimated solutions (on dense grids).

For example, this module contains functionality to compute off-grid marginals,
or to evaluate marginal likelihoods of observations of the solutions.
"""

import functools
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.backend import containers
from probdiffeq.strategies import smoothers

R = TypeVar("R")
"""Type-variable for random variables used in \
 generic initial value problem solutions."""


@jax.tree_util.register_pytree_node_class
class Solution(Generic[R]):
    """Estimated initial value problem solution."""

    def __init__(
        self,
        t,
        u,
        output_scale,
        marginals: R,
        posterior,
        num_steps,
    ):
        self.t = t
        self.u = u
        self.output_scale = output_scale
        self.marginals = marginals
        self.posterior = posterior
        self.num_steps = num_steps

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"t={self.t},"
            f"u={self.u},"
            f"output_scale={self.output_scale},"
            f"marginals={self.marginals},"
            f"posterior={self.posterior},"
            f"num_steps={self.num_steps},"
            ")"
        )

    def tree_flatten(self):
        children = (
            self.t,
            self.u,
            self.marginals,
            self.posterior,
            self.output_scale,
            self.num_steps,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        t, u, marginals, posterior, output_scale, n = children
        return cls(
            t=t,
            u=u,
            marginals=marginals,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=n,
        )

    def __len__(self):
        if jnp.ndim(self.t) < 1:
            raise ValueError("Solution object not batched :(")
        return self.t.shape[0]

    def __getitem__(self, item):
        if jnp.ndim(self.t) < 1:
            raise ValueError(f"Solution object not batched :(, {jnp.ndim(self.t)}")
        if isinstance(item, tuple) and len(item) > jnp.ndim(self.t):
            # s[2, 3] forbidden
            raise ValueError(f"Inapplicable shape: {item, jnp.shape(self.t)}")
        return Solution(
            t=self.t[item],
            u=self.u[item],
            output_scale=self.output_scale[item],
            # todo: make iterable?
            marginals=jax.tree_util.tree_map(lambda x: x[item], self.marginals),
            # todo: make iterable?
            posterior=jax.tree_util.tree_map(lambda x: x[item], self.posterior),
            num_steps=self.num_steps[item],
        )

    def __iter__(self):
        for i in range(self.t.shape[0]):
            yield self[i]


# todo: the functions in here should only depend on posteriors / strategies!


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
        return _offgrid_marginals(
            t=t, solution=s, solution_previous=sprev, solver=solver
        )

    return marginals_vmap(solution_left, ts, solution_right)


def _offgrid_marginals(*, solution, t, solution_previous, solver):
    return solver.strategy.offgrid_marginals(
        marginals=solution.marginals,
        posterior=solution.posterior,
        posterior_previous=solution_previous.posterior,
        t=t,
        t0=solution_previous.t,
        t1=solution.t,
        output_scale=solution.output_scale,
    )


def log_marginal_likelihood_terminal_values(*, observation_std, u, posterior, strategy):
    """Compute the log-marginal-likelihood of \
     observations of the IVP solution at the terminal value.

    Parameters
    ----------
    observation_std
        Standard deviation of the observation. Expected to be a scalar.
    u
        Observation. Expected to have shape (d,) for an ODE with shape (d,).
    posterior
        Posterior distribution.
        Expected to correspond to a solution of an ODE with shape (d,).
    strategy
        Strategy (that has been used to compute the solution).
        Expected to correspond to a solution of an ODE with shape (d,).
    """
    if jnp.shape(observation_std) != ():
        raise ValueError(
            "Scalar observation noise expected. "
            f"Shape {jnp.shape(observation_std)} received."
        )

    if jnp.ndim(u) >= 2:  # not valid for scalar or matrix-valued solutions
        raise ValueError(
            "Terminal-value solution (ndim=1, shape=(n,)) expected. "
            f"ndim={jnp.ndim(u)}, shape={jnp.shape(u)} received."
        )

    ssv = strategy.init(None, posterior).ssv
    obs, _ = ssv.observe_qoi(observation_std=observation_std)
    return jnp.sum(obs.logpdf(u))


def log_marginal_likelihood(*, observation_std, u, posterior, strategy):
    """Compute the log-marginal-likelihood of \
     observations of the IVP solution.

    Parameters
    ----------
    observation_std
        Standard deviation of the observation. Expected to be have shape (n,).
    u
        Observation. Expected to have shape (n, d) for an ODE with shape (d,).
    posterior
        Posterior distribution.
        Expected to correspond to a solution of an ODE with shape (d,).
    strategy
        Strategy (that has been used to compute the solution).
        Expected to correspond to a solution of an ODE with shape (d,).

    !!! note
        Use `log_marginal_likelihood_terminal_values`
        to compute the log-likelihood at the terminal values.

    """
    # todo: complain if it is used with a filter, not a smoother?
    # todo: allow option for log-posterior

    if not isinstance(posterior, smoothers.SmootherSol):
        msg1 = "Time-series marginal likelihoods "
        msg2 = "cannot be computed with a filtering solution."
        raise TypeError(msg1 + msg2)

    if jnp.shape(observation_std) != (jnp.shape(u)[0],):
        raise ValueError(
            f"Observation-noise shape {jnp.shape(observation_std)} does not match "
            f"the observation shape {jnp.shape(u)}. "
            f"Expected observation-noise shape: "
            f"{(jnp.shape(u)[0],)} != {jnp.shape(observation_std)}. "
        )

    if jnp.ndim(u) < 2:
        raise ValueError(
            "Time-series solution (ndim=2, shape=(n, m)) expected. "
            f"ndim={jnp.ndim(u)}, shape={jnp.shape(u)} received."
        )

    result = _kalman_filter(u, posterior.rand, observation_std, strategy=strategy)
    return result


# todo: this smells a lot like a `statespace.SSV` object.
#  But merging those two data structures might be in the far future.


class _KalFiltState(containers.NamedTuple):
    rv: Any
    num_data_points: int
    log_marginal_likelihood: float


# todo: this should return a Filtering posterior or a smoothing posterior
#  which could then be plotted. Right?
#  (We might also want some dense-output/checkpoint kind of thing here)
# todo: we should reuse the extrapolation model statespace.
#  But this only works if the ODE posterior uses the preconditioner (I think).
# todo: we should allow proper noise, and proper information functions.
#  But it is not clear which data structure that should be.
def _kalman_filter(u, /, mseq, standard_deviations, *, strategy, reverse=True):
    # the 0th backward model contains meaningless values
    bw_models = jax.tree_util.tree_map(lambda x: x[1:, ...], mseq.backward_model)

    # Incorporate final data point
    rv_terminal = jax.tree_util.tree_map(lambda x: x[-1, ...], mseq.init)
    init = _init_fn(rv_terminal, standard_deviations[-1], u[-1], strategy)

    # Scan over the remaining data points
    lml_state, _ = jax.lax.scan(
        f=functools.partial(_filter_step, strategy=strategy),
        init=init,
        xs=(bw_models, standard_deviations[:-1], u[:-1]),
        reverse=reverse,
    )
    return lml_state.log_marginal_likelihood


def _init_fn(rv, obs_std, data, strategy):
    ssv, _ = strategy.extrapolation.filter.init(rv)
    obs, cond_cor = ssv.observe_qoi(observation_std=obs_std)
    cor = cond_cor(data)
    lml_new = jnp.sum(obs.logpdf(data))
    return _KalFiltState(cor, 1.0, log_marginal_likelihood=lml_new)


def _filter_step(carry, x, *, strategy):
    # Read
    rv = carry.rv
    num_data = carry.num_data_points
    lml_prev = carry.log_marginal_likelihood
    bw_model, obs_std, data = x

    # Extrapolate
    rv_ext = bw_model.marginalise(rv)

    # Correct (with an alias for long function names)
    ssv, _ = strategy.extrapolation.filter.init(rv_ext)
    obs, cond_cor = ssv.observe_qoi(observation_std=obs_std)
    cor = cond_cor(data)

    # Compute marginal log likelihood (with an alias for long function names)
    lml_new = jnp.sum(obs.logpdf(data))
    lml_updated = (num_data * lml_prev + lml_new) / (num_data + 1)

    # Return values
    x = _KalFiltState(cor, num_data + 1, log_marginal_likelihood=lml_updated)
    return x, x
