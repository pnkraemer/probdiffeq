import jax

from probdiffeq.backend import containers


class Normal(containers.NamedTuple):
    mean: jax.Array
    cholesky: jax.Array
