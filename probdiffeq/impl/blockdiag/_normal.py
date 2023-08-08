import jax


class Normal:
    def __init__(self, mean, cholesky):
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
