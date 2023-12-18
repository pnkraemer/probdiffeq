"""Special functions."""

import jax.numpy as jnp
import scipy.special


def roots_hermitenorm(n, mu):
    pts, weights, sum_ = scipy.special.roots_hermitenorm(n=n, mu=mu)
    pts = jnp.asarray(pts)
    weights = jnp.asarray(weights)
    sum_ = jnp.asarray(sum_)
    return pts, weights, sum_
