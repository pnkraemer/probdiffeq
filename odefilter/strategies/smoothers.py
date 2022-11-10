"""Inference via smoothing."""

import abc
from typing import Any, Generic, TypeVar

import jax

from odefilter.strategies import _strategy

SSVTypeVar = TypeVar("SSVTypeVar")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class BackwardModel(Generic[SSVTypeVar]):
    """Backward model for backward-Gauss--Markov process representations."""

    def __init__(self, *, transition: Any, noise: SSVTypeVar):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[SSVTypeVar]):
    """Markov sequence."""

    def __init__(self, *, init: SSVTypeVar, backward_model: BackwardModel[SSVTypeVar]):
        self.init = init
        self.backward_model = backward_model

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(init={self.init}, backward_model={self.backward_model})"

    def tree_flatten(self):
        children = (self.init, self.backward_model)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, backward_model = children
        return cls(init=init, backward_model=backward_model)

    def scale_covariance(self, *, scale_sqrtm):
        noise = self.backward_model.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        bw_model = BackwardModel(transition=self.backward_model.transition, noise=noise)

        init = self.init.scale_covariance(scale_sqrtm=scale_sqrtm)
        return MarkovSequence(init=init, backward_model=bw_model)


class _SmootherCommon(_strategy.Strategy):
    """Common functionality for smoothers."""

    # Inherited abstract methods

    @abc.abstractmethod
    def case_interpolate(self, *, p0, rv1, t, t0, t1, scale_sqrtm):
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
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        raise NotImplementedError

    def init_posterior(self, *, taylor_coefficients):
        corrected = self.implementation.extrapolation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        backward_transition = (
            self.implementation.extrapolation.init_backward_transition()
        )
        backward_noise = self.implementation.extrapolation.init_backward_noise(
            rv_proto=corrected
        )
        backward_model = BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        posterior = MarkovSequence(init=corrected, backward_model=backward_model)
        return posterior

    def begin_extrapolation(self, *, posterior, dt):
        return self.implementation.extrapolation.begin_extrapolation(
            posterior.init.mean, dt=dt
        )

    def complete_correction(self, *, extrapolated, cache_obs):
        a, (corrected, b) = self.implementation.correction.complete_correction(
            extrapolated=extrapolated.init, cache=cache_obs
        )
        corrected_seq = MarkovSequence(
            init=corrected, backward_model=extrapolated.backward_model
        )
        return a, (corrected_seq, b)

    def extract_sol_terminal_value(self, *, posterior):
        return posterior.init.extract_qoi()

    def marginals_terminal_value(self, *, posterior):
        return posterior.init

    def marginals(self, *, posterior):
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)
        marginals = self.implementation.extrapolation.marginalise_backwards(
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
        samples = self.implementation.extrapolation.sample_backwards(
            init, posterior.backward_model.transition, noise, base_samples
        )
        u = noise.extract_qoi_from_sample(samples)
        return u, samples

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale_sqrtm, t, t0):
        dt = t - t0
        linearisation_pt, cache = self.implementation.extrapolation.begin_extrapolation(
            rv.mean, dt=dt
        )
        extrapolated, (
            bw_noise,
            bw_op,
        ) = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            l0=rv.cov_sqrtm_lower,
            output_scale_sqrtm=output_scale_sqrtm,
            cache=cache,
        )
        backward_model = BackwardModel(transition=bw_op, noise=bw_noise)
        return extrapolated, backward_model  # should this return a MarkovSequence?

    def _duplicate_with_unit_backward_model(self, *, posterior):
        bw_transition0 = self.implementation.extrapolation.init_backward_transition()
        bw_noise0 = self.implementation.extrapolation.init_backward_noise(
            rv_proto=posterior.backward_model.noise
        )
        bw_model = BackwardModel(transition=bw_transition0, noise=bw_noise0)

        return MarkovSequence(init=posterior.init, backward_model=bw_model)


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        extrapolated, (
            bw_noise,
            bw_op,
        ) = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            l0=posterior_previous.init.cov_sqrtm_lower,
            cache=cache,
            output_scale_sqrtm=output_scale_sqrtm,
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
        marginals_t = self.implementation.extrapolation.marginalise_model(
            init=marginals,
            linop=acc.backward_model.transition,
            noise=acc.backward_model.noise,
        )
        u = marginals_t.extract_qoi()
        return u, marginals_t


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_SmootherCommon):
    """Fixed-point smoother."""

    def complete_extrapolation(
        self, linearisation_pt, cache, *, posterior_previous, output_scale_sqrtm
    ):
        extrapolated, (
            bw_noise,
            bw_op,
        ) = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            l0=posterior_previous.init.cov_sqrtm_lower,
            output_scale_sqrtm=output_scale_sqrtm,
            cache=cache,
        )
        bw_increment = BackwardModel(transition=bw_op, noise=bw_noise)

        noise, gain = self.implementation.extrapolation.condense_backward_models(
            transition_state=bw_increment.transition,
            noise_state=bw_increment.noise,
            transition_init=posterior_previous.backward_model.transition,
            noise_init=posterior_previous.backward_model.noise,
        )
        backward_model = BackwardModel(transition=gain, noise=noise)

        return MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):  # s1.t == t

        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        noise0, g0 = self.implementation.extrapolation.condense_backward_models(
            transition_state=p1.backward_model.transition,
            noise_state=p1.backward_model.noise,
            transition_init=p0.backward_model.transition,
            noise_init=p0.backward_model.noise,
        )
        backward_model1 = BackwardModel(transition=g0, noise=noise0)

        solution = MarkovSequence(init=p1.init, backward_model=backward_model1)
        accepted = self._duplicate_with_unit_backward_model(posterior=solution)
        previous = accepted

        return accepted, solution, previous

    def case_interpolate(self, *, p0, rv1, t, t0, t1, scale_sqrtm):
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
        noise0, g0 = self.implementation.extrapolation.condense_backward_models(
            transition_state=bw0.transition,
            noise_state=bw0.noise,
            transition_init=p0.backward_model.transition,
            noise_init=p0.backward_model.noise,
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
