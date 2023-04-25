"""Markov sequences and Markov processes."""

import functools
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.backend import control_flow

S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""

# todo: markov sequences should not necessarily be backwards


@jax.tree_util.register_pytree_node_class
class MarkovSequence(Generic[S]):
    """Markov sequence. A discretised Markov process."""

    def __init__(self, *, init: S, backward_model):
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

    def sample(self, key, *, shape):
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        base_samples = jax.random.normal(key=key, shape=shape + self.sample_shape)
        return self._transform_unit_sample(base_samples)

    def _transform_unit_sample(self, base_sample, /):
        if base_sample.shape == self.sample_shape:
            return self._transform_one_unit_sample(base_sample)

        transform_vmap = jax.vmap(self._transform_unit_sample, in_axes=0)
        return transform_vmap(base_sample)

    def _transform_one_unit_sample(self, base_sample, /):
        def body_fun(carry, conditionals_and_base_samples):
            _, samp_prev = carry
            conditional, base = conditionals_and_base_samples

            cond = conditional(samp_prev)
            samp = cond.transform_unit_sample(base)
            qoi = cond.extract_qoi_from_sample(samp)

            return (qoi, samp), (qoi, samp)

        # Compute a sample at the terminal value
        init = jax.tree_util.tree_map(lambda s: s[-1, ...], self.init)
        init_sample = init.transform_unit_sample(base_sample[-1])
        init_qoi = init.extract_qoi_from_sample(init_sample)
        init_val = (init_qoi, init_sample)

        # Remove the initial backward-model
        conds = jax.tree_util.tree_map(lambda s: s[1:, ...], self.backward_model)

        # Loop over backward models and the remaining base samples
        xs = (conds, base_sample[:-1])
        _, (qois, samples) = jax.lax.scan(
            f=body_fun, init=init_val, xs=xs, reverse=True
        )
        qois_full = jnp.concatenate((qois, init_qoi[None, ...]))
        samples_full = jnp.concatenate((samples, init_sample[None, ...]))
        return qois_full, samples_full

    def marginalise_backwards(self):
        def body_fun(rv, conditional):
            out = conditional.marginalise(rv)
            return out, out

        # Initial backward model leads into the void
        # todo: this is only true for the version we use.
        conds = jax.tree_util.tree_map(lambda x: x[1:, ...], self.backward_model)

        # If we hold many 'init's, choose the terminal one.
        if self.backward_model.noise.mean.shape == self.init.mean.shape:
            init = jax.tree_util.tree_map(lambda x: x[-1, ...], self.init)
        else:
            init = self.init

        # Scan and return
        reverse_scan = functools.partial(control_flow.scan_with_init, reverse=True)
        _, rvs = reverse_scan(f=body_fun, init=init, xs=conds)
        return rvs

    @property
    def sample_shape(self):
        return self.backward_model.noise.sample_shape
