"""Time-stepping utilities."""

import jax.numpy as jnp


def initial(vf_autonomous, initial_values, /, scale=0.01, nugget=1e-5):
    """Propose an initial time-step."""
    u0, *_ = initial_values
    f0 = vf_autonomous(*initial_values)

    norm_y0 = jnp.linalg.norm(u0)
    norm_dy0 = jnp.linalg.norm(f0) + nugget

    return scale * norm_y0 / norm_dy0


def initial_adaptive(vf, initial_values, /, t0, *, error_contraction_rate, rtol, atol):
    """Propose an initial time-step as a function of the tolerances."""
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    # Implementation mostly copied from
    #
    # https://github.com/google/jax/blob/main/jax/experimental/ode.py
    #

    if len(initial_values) > 1:
        raise ValueError
    y0 = initial_values[0]

    f0 = vf(y0, t=t0)
    scale = atol + jnp.abs(y0) * rtol
    d0, d1 = jnp.linalg.norm(y0), jnp.linalg.norm(f0)

    dt0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + dt0 * f0
    f1 = vf(y1, t=t0 + dt0)
    d2 = jnp.linalg.norm((f1 - f0) / scale) / dt0

    dt1 = jnp.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        jnp.maximum(1e-6, dt0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (error_contraction_rate + 1.0)),
    )
    return jnp.minimum(100.0 * dt0, dt1)
