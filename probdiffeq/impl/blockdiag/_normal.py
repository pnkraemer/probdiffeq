import jax


class Normal:
    def __init__(self, mean, cholesky):
        # # Only shape-check if the inputs are arrays.
        # # When debugging, for example, mean and cholesky may be shapes or ndims
        # if isinstance(mean, jax.Array):
        #     assert isinstance(cholesky, jax.Array)
        #     *b1, n1 = jnp.shape(mean)
        #     *b2, n2, n3 = jnp.shape(cholesky)
        #     assert b1 == b2 and n1 == n2 == n3
        #
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
