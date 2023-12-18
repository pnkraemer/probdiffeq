"""Special functions."""

import jax.scipy.special


def roots_hermitenorm(n, mu):
    return jax.scipy.special.roots_hermitenorm(n=n, mu=mu)
