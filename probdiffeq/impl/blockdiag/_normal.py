import jax
import jax.numpy as jnp


class Normal:
    def __init__(self, mean, cholesky):
        d, n = jnp.shape(mean)
        assert jnp.shape(cholesky) == (d, n, n)

        self.mean = mean
        self.cholesky = cholesky


jax.tree_util.register_pytree_node(
    Normal,
    flatten_func=lambda n: ((n.mean, n.cholesky), ()),
    unflatten_func=lambda _a, c: Normal(*c),
)
