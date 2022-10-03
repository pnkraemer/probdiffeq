"""Utilities for working with posteriors."""

from functools import partial

import jax

from odefilter.blox import sqrtutil


@partial(jax.jit, static_argnames=("reverse",))
def marginalize(*, m0, c_sqrtm0, A, b, Q_sqrtm, reverse):
    """Compute the marginals of a linear model."""

    @jax.jit
    def body_fn(s, model):
        a, x, d_sqrtm = model
        m, c_sqrtm = s
        m_new = a @ m + x
        c_sqrtm_new = sqrtutil.sum_of_sqrtm_factors(S1=a @ c_sqrtm, S2=d_sqrtm)
        return (m_new, c_sqrtm_new), (m_new, c_sqrtm_new)

    return jax.lax.scan(
        f=body_fn, init=(m0, c_sqrtm0), xs=(A, b, Q_sqrtm), reverse=reverse
    )
