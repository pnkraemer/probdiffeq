"""Inference via smoothing."""

import abc
from typing import Any, Generic, TypeVar

import jax
import jax.tree_util

from odefilter.strategies import _strategy

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
class BackwardModel(Generic[T]):
    """Backward model for backward-Gauss--Markov process representations."""

    def __init__(self, *, transition: Any, noise: T):
        self.transition = transition
        self.noise = noise

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[T]):
    """Markov sequence."""

    def __init__(self, *, init: T, backward_model: BackwardModel[T]):
        self.init = init
        self.backward_model = backward_model

    def tree_flatten(self):
        children = (self.init, self.backward_model)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, backward_model = children
        return cls(init=init, backward_model=backward_model)


class _SmootherCommon(_strategy.Strategy):
    """Common functionality for smoothers."""

    # Inherited abstract methods

    @abc.abstractmethod
    def case_interpolate(self, *, p0, rv1, t, t0, t1, scale_sqrtm):  # noqa: D102
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(
        self, *, t, marginals, posterior_previous, t0, t1, scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, m_ext, cache, *, output_scale_sqrtm, p, p_inv, posterior_previous
    ):
        raise NotImplementedError

    def init_posterior(self, *, corrected):
        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        posterior = MarkovSequence(init=corrected, backward_model=backward_model)
        return posterior

    def extrapolate_mean(self, *, posterior, p_inv, p):
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            posterior.init.mean, p=p, p_inv=p_inv
        )
        return m_ext, (m_ext_p, m0_p)

    def final_correction(self, *, extrapolated, cache_obs, m_obs):
        a, (corrected, b) = self.implementation.final_correction(
            extrapolated=extrapolated.init, cache_obs=cache_obs, m_obs=m_obs
        )
        corrected_seq = MarkovSequence(
            init=corrected, backward_model=extrapolated.backward_model
        )
        return a, (corrected_seq, b)

    def extract_sol_terminal_value(self, *, posterior):
        return self.implementation.extract_sol(rv=posterior.init)

    def extract_sol_from_marginals(self, *, marginals):
        return self.implementation.extract_sol(rv=marginals)

    def marginals_terminal_value(self, *, posterior):
        return posterior.init

    def marginals(self, *, posterior):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        marginals = self.implementation.marginalise_backwards(
            init=init,
            linop=posterior.backward_model.transition,
            noise=posterior.backward_model.noise,
        )
        return marginals

    def sample(self, key, *, posterior, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        sample_shape = posterior.backward_model.noise.mean.shape
        base_samples = self._base_samples(key, shape=shape + sample_shape)
        return self.transform_base_samples(
            posterior=posterior, base_samples=base_samples
        )

    def transform_base_samples(self, posterior, base_samples):
        if base_samples.ndim == posterior.backward_model.noise.mean.ndim:
            return self._sample_one(posterior, base_samples)

        transform_vmap = jax.vmap(self.transform_base_samples, in_axes=(None, 0))
        return transform_vmap(posterior, base_samples)

    def _sample_one(self, posterior, base_samples):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        noise = posterior.backward_model.noise
        samples = self.implementation.sample_backwards(
            init, posterior.backward_model.transition, noise, base_samples
        )
        u = self.implementation.extract_mean_from_marginals(samples)
        return u, samples

    def scale_marginals(self, marginals, *, output_scale_sqrtm):
        return self.implementation.scale_covariance(
            rv=marginals, scale_sqrtm=output_scale_sqrtm
        )

    def scale_posterior(self, posterior, *, output_scale_sqrtm):
        init = self.implementation.scale_covariance(
            rv=posterior.init, scale_sqrtm=output_scale_sqrtm
        )
        noise = self.implementation.scale_covariance(
            rv=posterior.backward_model.noise, scale_sqrtm=output_scale_sqrtm
        )

        bw_model = BackwardModel(
            transition=posterior.backward_model.transition, noise=noise
        )
        return MarkovSequence(init=init, backward_model=bw_model)

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale_sqrtm, t, t0):
        dt = t - t0
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            rv.mean, p=p, p_inv=p_inv
        )

        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            m_ext_p=m_ext_p,
            m0_p=m0_p,
            l0=rv.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
        return extrapolated, backward_model  # should this return a MarkovSequence?

    def _duplicate_with_unit_backward_model(self, *, posterior):
        bw_transition0 = self.implementation.init_backward_transition()
        bw_noise0 = self.implementation.init_backward_noise(
            rv_proto=posterior.backward_model.noise
        )
        bw_model = BackwardModel(transition=bw_transition0, noise=bw_noise0)

        return MarkovSequence(init=posterior.init, backward_model=bw_model)


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def complete_extrapolation(
        self, m_ext, cache, *, output_scale_sqrtm, p, p_inv, posterior_previous
    ):
        m_ext_p, m0_p = cache
        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=posterior_previous.init.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
        return MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):  # s1.t == t

        # todo: is this duplication unnecessary?
        accepted = self._duplicate_with_unit_backward_model(posterior=p1)

        return accepted, p1, p1

    def case_interpolate(self, *, p0, rv1, t0, t1, t, scale_sqrtm):
        # A smoother interpolates by reverting the Markov kernels between s0.t and t
        # which gives an extrapolation and a backward transition;
        # and by reverting the Markov kernels between t and s1.t
        # which gives another extrapolation and a backward transition.
        # The latter extrapolation is discarded in favour of s1.marginals_filtered,
        # but the backward transition is kept.

        # rv0, diffsqrtm = p0.init, s1.output_scale_sqrtm

        # Extrapolate from t0 to t, and from t to t1
        extrapolated0, backward_model0 = self._interpolate_from_to_fn(
            rv=p0.init, output_scale_sqrtm=scale_sqrtm, t=t, t0=t0
        )
        posterior0 = MarkovSequence(init=extrapolated0, backward_model=backward_model0)

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=scale_sqrtm, t=t1, t0=t
        )
        posterior1 = MarkovSequence(init=rv1, backward_model=backward_model1)

        return posterior1, posterior0, posterior0

    def offgrid_marginals(
        self, *, t, marginals, posterior_previous, t0, t1, scale_sqrtm
    ):
        acc, _sol, _prev = self.case_interpolate(
            t=t,
            rv1=marginals,
            p0=posterior_previous,
            t0=t0,
            t1=t1,
            scale_sqrtm=scale_sqrtm,
        )
        # todo: what to do here? We need to smooth from the marginals,
        # but we only get the posterior. Right?
        marginals_t = self.implementation.marginalise_model(
            init=marginals,
            linop=acc.backward_model.transition,
            noise=acc.backward_model.noise,
        )
        u = self.extract_sol_from_marginals(marginals=marginals_t)
        return u, marginals_t


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_SmootherCommon):
    """Fixed-point smoother."""

    def complete_extrapolation(
        self, m_ext, cache, *, posterior_previous, output_scale_sqrtm, p, p_inv
    ):
        m_ext_p, m0_p = cache
        extrapolated, (bw_noise, bw_op) = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=posterior_previous.init.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        bw_increment = BackwardModel(transition=bw_op, noise=bw_noise)

        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment,
            bw_init=posterior_previous.backward_model,
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        return MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):  # s1.t == t

        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=p0.backward_model, bw_state=p1.backward_model
        )
        backward_model1 = BackwardModel(transition=g0, noise=noise0)

        solution = MarkovSequence(init=p1.init, backward_model=backward_model1)
        accepted = self._duplicate_with_unit_backward_model(posterior=solution)
        previous = accepted

        return accepted, solution, previous

    def case_interpolate(self, *, p0, rv1, t, t0, t1, scale_sqrtm):  # noqa: D102
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=p0.init,
            output_scale_sqrtm=scale_sqrtm,
            t=t,
            t0=t0,
        )
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=p0.backward_model, bw_state=bw0
        )
        backward_model0 = BackwardModel(transition=g0, noise=noise0)
        solution = MarkovSequence(init=extrapolated0, backward_model=backward_model0)

        previous = self._duplicate_with_unit_backward_model(posterior=solution)

        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=scale_sqrtm, t=t1, t0=t
        )
        accepted = MarkovSequence(init=rv1, backward_model=backward_model1)
        return accepted, solution, previous

    def offgrid_marginals(
        self, *, t, marginals, posterior_previous, t0, t1, scale_sqrtm
    ):
        raise NotImplementedError
