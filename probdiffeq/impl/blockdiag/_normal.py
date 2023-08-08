import jax
import jax.numpy as jnp


class Normal:
    def __init__(self, mean, cholesky):
        # Only shape-check if the inputs are arrays.
        # When debugging, for example, mean and cholesky may be shapes or ndims
        if isinstance(mean, jax.Array):
            assert isinstance(cholesky, jax.Array)
            assert jnp.ndim(mean) >= 2, jnp.shape(mean)
            assert jnp.ndim(cholesky) >= 3, jnp.shape(cholesky)

            *b1, d1, n1 = jnp.shape(mean)
            *b2, d2, n2, n3 = jnp.shape(cholesky)

            print_if_fails = (jnp.shape(mean), jnp.shape(cholesky))
            assert b1 == b2 and d1 == d2 and n1 == n2 == n3, print_if_fails

        self.mean = mean
        self.cholesky = cholesky


def _flatten(normal):
    children = (normal.mean, normal.cholesky)
    aux = ()
    return children, aux


def _unflatten(_aux, children):
    (mean, cholesky) = children
    return Normal(mean, cholesky)


jax.tree_util.register_pytree_node(Normal, _flatten, _unflatten)
