"""''Global'' estimation: smoothing."""

import abc
import functools
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import _control_flow
from probdiffeq.strategies import _strategy

SSVTypeVar = TypeVar("SSVTypeVar")
"""A type-variable to alias appropriate state-space variable types."""

# todo: markov sequences should not necessarily be backwards


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[SSVTypeVar]):
    """Markov sequence. A discretised Markov process."""

    def __init__(self, *, init: SSVTypeVar, backward_model):
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

    def transform_unit_sample(self, base_sample, /):
        if base_sample.shape == self.sample_shape:
            return self._transform_one_unit_sample(base_sample)

        transform = self.transform_unit_sample
        transform_vmap = jax.vmap(transform, in_axes=0)
        return transform_vmap(base_sample)

    def _transform_one_unit_sample(self, base_sample, /):
        def body_fun(carry, conditionals_and_base_samples):
            _, samp_prev = carry
            conditional, base = conditionals_and_base_samples

            cond = conditional(samp_prev)
            samp = cond.hidden_state.transform_unit_sample(base)
            qoi = cond.extract_qoi_from_sample(samp)

            return (qoi, samp), (qoi, samp)

        # Compute a sample at the terminal value
        init = jax.tree_util.tree_map(lambda s: s[-1, ...], self.init)
        init_sample = init.hidden_state.transform_unit_sample(base_sample[-1])
        init_qoi = init.extract_qoi_from_sample(init_sample)
        init_val = (init_qoi, init_sample)

        # Remove the initial backward-model
        conds = jax.tree_util.tree_map(lambda s: s[1:, ...], self.backward_model)

        # Loop over backward models and the remaining base samples
        xs = (conds, base_sample[:-1])
        _, (qois, samples) = jax.lax.scan(
            f=body_fun, init=init_val, xs=xs, reverse=True
        )
        qois_full = jnp.vstack((qois, init_qoi[None, ...]))
        samples_full = jnp.vstack((samples, init_sample[None, ...]))
        return qois_full, samples_full

    def marginalise_backwards(self):
        def body_fun(rv, conditional):
            out = conditional.marginalise(rv)
            return out, out

        # Initial condition does not matter
        conds = jax.tree_util.tree_map(lambda x: x[1:, ...], self.backward_model)

        # Scan and return
        reverse_scan = functools.partial(_control_flow.scan_with_init, reverse=True)
        _, rvs = reverse_scan(f=body_fun, init=self.init, xs=conds)
        return rvs

    @property
    def sample_shape(self):
        return self.backward_model.noise.sample_shape


class _SmootherCommon(_strategy.Strategy):
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
        bw_model = init_bw_model(ssv_proto=corrected)
        return MarkovSequence(init=corrected, backward_model=bw_model)

    def begin_extrapolation(self, *, posterior, dt):
        return self.implementation.extrapolation.begin_extrapolation(
            posterior.init, dt=dt
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
        markov = MarkovSequence(init=init, backward_model=posterior.backward_model)
        return markov.marginalise_backwards()

    def sample(self, key, *, posterior, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        base_samples = jax.random.normal(key=key, shape=shape + posterior.sample_shape)
        return posterior.transform_unit_sample(base_samples)

    # Auxiliary routines that are the same among all subclasses

    def _interpolate_from_to_fn(self, *, rv, output_scale_sqrtm, t, t0):
        dt = t - t0
        linearisation_pt, cache = self.implementation.extrapolation.begin_extrapolation(
            rv, dt=dt
        )
        extrapolated, bw_model = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            p0=rv,
            output_scale_sqrtm=output_scale_sqrtm,
            cache=cache,
        )
        return extrapolated, bw_model  # should this return a MarkovSequence?

    # todo: should this be a classmethod of MarkovSequence?
    def _duplicate_with_unit_backward_model(self, *, posterior):
        init_conditional_fn = self.implementation.extrapolation.init_conditional
        bw_model = init_conditional_fn(ssv_proto=posterior.init)
        return MarkovSequence(init=posterior.init, backward_model=bw_model)


@jax.tree_util.register_pytree_node_class
class Smoother(_SmootherCommon):
    """Smoother."""

    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        extrapolated, bw_model = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            p0=posterior_previous.init,
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
        marginals_at_t = acc.backward_model.marginalise(marginals)
        u = marginals_at_t.extract_qoi()
        return u, marginals_at_t


@jax.tree_util.register_pytree_node_class
class FixedPointSmoother(_SmootherCommon):
    """Fixed-point smoother.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def complete_extrapolation(
        self, linearisation_pt, cache, *, posterior_previous, output_scale_sqrtm
    ):
        _temp = self.implementation.extrapolation.revert_markov_kernel(
            linearisation_pt=linearisation_pt,
            p0=posterior_previous.init,
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
