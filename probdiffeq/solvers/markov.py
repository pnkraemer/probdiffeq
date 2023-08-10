"""Markov sequences and Markov processes."""

from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.impl import impl

S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""

R = TypeVar("R", bound=bool)
"""A (Boolean-bound) type-variable indicating the direction of the Markov sequence."""


# todo: Unify the MarkovSeq* implementations (Naming, use of preconditioner, fwd/rev)


@jax.tree_util.register_pytree_node_class
class MarkovSeqPreconFwd(Generic[S]):
    def __init__(self, *, init: S, conditional, preconditioner):
        self.init = init
        self.conditional = conditional
        self.preconditioner = preconditioner

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"init={self.init}, conditional={self.conditional}"
        args2 = f"preconditioner={self.preconditioner}"
        return f"{name}({args1}, {args2})"

    def tree_flatten(self):
        children = (self.init, self.conditional, self.preconditioner)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, conditional, preconditioner = children
        return cls(init=init, conditional=conditional, preconditioner=preconditioner)


@jax.tree_util.register_pytree_node_class
class MarkovSeqPreconRev(Generic[S]):
    def __init__(self, *, init: S, conditional, preconditioner):
        self.init = init
        self.conditional = conditional
        self.preconditioner = preconditioner

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"init={self.init}, conditional={self.conditional}"
        args2 = f"preconditioner={self.preconditioner}"
        return f"{name}({args1}, {args2})"

    def tree_flatten(self):
        children = (self.init, self.conditional, self.preconditioner)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, conditional, preconditioner = children
        return cls(init=init, conditional=conditional, preconditioner=preconditioner)


def rescale_cholesky(markov_seq, factor):
    init = impl.random.rescale_cholesky(markov_seq.init, factor)
    A, noise = markov_seq.conditional
    noise = impl.random.rescale_cholesky(noise, factor)
    return MarkovSeqRev(init=init, conditional=(A, noise))


@jax.tree_util.register_pytree_node_class
class MarkovSeqRev(Generic[S]):
    """Markov sequence. A discretised Markov process."""

    def __init__(self, *, init: S, conditional):
        self.init = init
        self.conditional = conditional

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(init={self.init}, conditional={self.conditional})"

    def tree_flatten(self):
        children = (self.init, self.conditional)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        init, conditional = children
        return cls(init=init, conditional=conditional)

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

            rv = impl.conditional.apply(samp_prev, conditional)
            sample = impl.random.transform_unit_sample(base, rv)
            qoi = impl.random.qoi_from_sample(sample)
            return (qoi, sample), (qoi, sample)

        # Compute a sample at the terminal value
        init = jax.tree_util.tree_map(lambda s: s[-1, ...], self.init)
        init_sample = impl.random.transform_unit_sample(base_sample[-1], init)
        init_qoi = impl.random.qoi_from_sample(init_sample)
        init_val = (init_qoi, init_sample)

        # Loop over backward models and the remaining base samples
        xs = (self.conditional, base_sample[:-1])
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

        # If we hold many 'init's, choose the terminal one.
        if self.conditional.noise.mean.shape == self.init.mean.shape:
            init = jax.tree_util.tree_map(lambda x: x[-1, ...], self.init)
        else:
            init = self.init

        # Scan and return
        _, rvs = jax.lax.scan(f=body_fun, init=init, xs=self.conditional, reverse=True)
        return rvs

    @property
    def sample_shape(self):
        # The number of samples is one larger than the number of conditionals
        _, noise = self.conditional
        n, *shape_single_sample = impl.random.sample_shape(noise)
        return (n + 1, *tuple(shape_single_sample))

    def scale_covariance(self, output_scale):
        init = self.init.scale_covariance(output_scale)
        conditional = self.conditional.scale_covariance(output_scale)
        return MarkovSeqRev(init=init, conditional=conditional)


def marginals(markov_seq: MarkovSeqRev):
    def step(x, cond):
        extrapolated = impl.conditional.marginalise(x, cond)
        return extrapolated, extrapolated

    # If we hold many 'init's, choose the terminal one.
    _, noise = markov_seq.conditional
    if noise.mean.shape == markov_seq.init.mean.shape:
        print("yes")
        init = jax.tree_util.tree_map(lambda x: x[-1, ...], markov_seq.init)
    else:
        print("no")
        init = markov_seq.init

    _, marg = jax.lax.scan(step, init=init, xs=markov_seq.conditional, reverse=True)
    return marg
