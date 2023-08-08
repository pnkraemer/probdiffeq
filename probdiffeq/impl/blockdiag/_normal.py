import jax
import jax.numpy as jnp


class Normal:
    def __init__(self, mean, cholesky):
        assert jnp.ndim(mean) >= 2, jnp.shape(mean)
        assert jnp.ndim(cholesky) >= 3, jnp.shape(cholesky)

        *_, d1, n1 = jnp.shape(mean)
        *_, d2, n2, n3 = jnp.shape(cholesky)
        assert d1 == d2 and n1 == n2 == n3, (jnp.shape(mean), jnp.shape(cholesky))

        self.mean = mean
        self.cholesky = cholesky


def _flatten(normal):
    children = (normal.mean, normal.cholesky)
    aux = ()
    return children, aux


def _unflatten(_aux, children):
    mean, cholesky = children
    return Normal(mean, cholesky)


jax.tree_util.register_pytree_node(Normal, _flatten, _unflatten)
