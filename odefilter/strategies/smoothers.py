"""Inference via smoothing."""

import abc
import functools
from typing import Any, Generic, TypeVar

import jax

from odefilter import _control_flow
from odefilter.strategies import _strategy

SSVTypeVar = TypeVar("SSVTypeVar")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[SSVTypeVar]):
    """Markov sequence."""

    def __init__(self, *, init: SSVTypeVar, backward_model: Any):
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
        bw_model = self.backward_model.scale_covariance(scale_sqrtm=scale_sqrtm)
        init = self.init.scale_covariance(scale_sqrtm=scale_sqrtm)
        return MarkovSequence(init=init, backward_model=bw_model)

    def transform_unit_sample(self, x, /):
        if x.ndim == self.backward_model.noise.mean.ndim:
            return self._transform_one_unit_sample(x)

        transform = self.transform_unit_sample
        transform_vmap = jax.vmap(transform, in_axes=0)
        return transform_vmap(x)

    def _transform_one_unit_sample(self, x, /):
        init = jax.tree_util.tree_map(lambda s: s[-1, ...], self.init)
        linop, noise = self.backward_model.transition, self.backward_model.noise
        linop_, noise_ = jax.tree_util.tree_map(lambda s: s[1:, ...], (linop, noise))

        noise_sample = noise_.transform_unit_sample(x[:-1])
        init_sample = init.transform_unit_sample(x[-1])
        init_qoi = init.extract_qoi_from_sample(init_sample)

        def body_fun(carry, op_and_noi):
            _, samp_last = carry
            op, noi = op_and_noi

            # todo: move the function below to the random variable implementations?
            samp = init.Ax_plus_y(A=op, x=samp_last, y=noi)
            qoi = init.extract_qoi_from_sample(samp)

            return (qoi, samp), (qoi, samp)

        xs = (linop_, noise_sample)
        init_val = (init_qoi, init_sample)
        reverse_scan = functools.partial(_control_flow.scan_with_init, reverse=True)
        _, (qois, samples) = reverse_scan(f=body_fun, init=init_val, xs=xs)
        return qois, samples


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

        init_bw_model = self.implementation.extrapolation.init_conditional
        bw_model = init_bw_model(rv_proto=corrected)
        return MarkovSequence(init=corrected, backward_model=bw_model)

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

        marginalise_fn = self.implementation.extrapolation.marginalise_backwards
        return marginalise_fn(init=init, conditionals=posterior.backward_model)

    def sample(self, key, *, posterior, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        sample_shape = posterior.backward_model.noise.mean.shape
        base_samples = self._base_samples(key, shape=shape + sample_shape)
        return posterior.transform_unit_sample(base_samples)

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale_sqrtm, t, t0):
        dt = t - t0
        linearisation_pt, cache = self.implementation.extrapolation.begin_extrapolation(
            rv.mean, dt=dt
        )
        extrapolated, bw_model = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            l0=rv.cov_sqrtm_lower,
            output_scale_sqrtm=output_scale_sqrtm,
            cache=cache,
        )
        return extrapolated, bw_model  # should this return a MarkovSequence?

    def _duplicate_with_unit_backward_model(self, *, posterior):
        bw_model = self.implementation.extrapolation.init_conditional(
            rv_proto=posterior.backward_model.noise
        )
        return MarkovSequence(init=posterior.init, backward_model=bw_model)


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        extrapolated, bw_model = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            l0=posterior_previous.init.cov_sqrtm_lower,
            cache=cache,
            output_scale_sqrtm=output_scale_sqrtm,
        )
        return MarkovSequence(init=extrapolated, backward_model=bw_model)

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
        _temp = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            l0=posterior_previous.init.cov_sqrtm_lower,
            output_scale_sqrtm=output_scale_sqrtm,
            cache=cache,
        )
        extrapolated, bw_increment = _temp

        merge_fn = posterior_previous.backward_model.merge_with_incoming_conditional
        backward_model = merge_fn(bw_increment)

        return MarkovSequence(init=extrapolated, backward_model=backward_model)

    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):  # s1.t == t

        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?
        merge_fn = p0.backward_model.merge_with_incoming_conditional
        backward_model1 = merge_fn(p1.backward_model)

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
        backward_model0 = p0.backward_model.merge_with_incoming_conditional(bw0)
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
