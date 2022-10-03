"""Step-size selection."""

import abc
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


def pi_control(*, atol, rtol, error_order):
    return _PIControl(), _PIControlParams(atol=atol, rtol=rtol, error_order=error_order)


class _PIControlParams(NamedTuple):

    atol: float
    rtol: float

    error_order: int

    safety = 0.95
    factor_min = 0.2
    factor_max = 10.0
    power_integral_unscaled = 0.3
    power_proportional_unscaled = 0.4


class _PIControl:
    def propose_first_dt(self, f, u0, params):
        return _propose_first_dt_per_tol(
            f=f,
            u0=u0,
            num_derivatives=params.error_order - 1,
            atol=params.atol,
            rtol=params.rtol,
        )

    def normalise_error(self, *, error, params, u1_ref):
        error_rel = error / (params.atol + params.rtol * u1_ref)
        error_norm = jnp.linalg.norm(error_rel) / jnp.sqrt(error.size)
        return error_norm

    def scale_factor(self, *, error_norm, error_norm_previously_accepted, params):
        return _scale_factor_pi_control(
            error_norm=error_norm,
            error_norm_previously_accepted=error_norm_previously_accepted,
            safety=params.safety,
            error_order=params.error_order,
            factor_min=params.factor_min,
            factor_max=params.factor_max,
            power_integral_unscaled=params.power_integral_unscaled,
            power_proportional_unscaled=params.power_proportional_unscaled,
        )


@partial(jax.jit, static_argnames=("f",))
def _propose_first_dt(*, f, u0):
    norm_y0 = jnp.linalg.norm(u0)
    norm_dy0 = jnp.linalg.norm(f(u0))
    return 0.01 * norm_y0 / norm_dy0


@partial(jax.jit, static_argnames=("f",))
def _propose_first_dt_per_tol(*, f, u0, num_derivatives, rtol, atol):
    # Taken from:
    # https://github.com/google/jax/blob/main/jax/experimental/ode.py
    #
    # which uses the algorithm from
    #
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    f0 = f(u0)
    scale = atol + u0 * rtol
    a = jnp.linalg.norm(u0 / scale)
    b = jnp.linalg.norm(f0 / scale)
    dt0 = jnp.where((a < 1e-5) | (b < 1e-5), 1e-6, 0.01 * a / b)

    u1 = u0 + dt0 * f0
    f1 = f(u1)
    c = jnp.linalg.norm((f1 - f0) / scale) / dt0
    dt1 = jnp.where(
        (b <= 1e-15) & (c <= 1e-15),
        jnp.maximum(1e-6, dt0 * 1e-3),
        (0.01 / jnp.max(b + c)) ** (1.0 / (num_derivatives + 1)),
    )
    return jnp.minimum(100.0 * dt0, dt1)


@jax.jit
def _scale_factor_integral_control(
    *, error_norm, safety, error_order, factor_min, factor_max
):
    """Integral control."""
    scale_factor = safety * (error_norm ** (-1.0 / error_order))
    scale_factor_clipped = jnp.maximum(
        factor_min, jnp.minimum(scale_factor, factor_max)
    )
    return scale_factor_clipped


@jax.jit
def _scale_factor_pi_control(
    *,
    error_norm,
    error_norm_previously_accepted,
    error_order,
    safety,
    factor_min,
    factor_max,
    power_integral_unscaled,
    power_proportional_unscaled,
):
    """Proportional-integral control.

    Proportional-integral control simplifies to integral control
    when the parameters are chosen as

        `power_integral_unscaled=1`,
        `power_proportional_unscaled=0`.
    """
    n1 = power_integral_unscaled / error_order
    n2 = power_proportional_unscaled / error_order

    a1 = (1.0 / error_norm) ** n1
    a2 = (error_norm_previously_accepted / error_norm) ** n2
    scale_factor = safety * a1 * a2

    scale_factor_clipped = jnp.maximum(
        factor_min, jnp.minimum(scale_factor, factor_max)
    )
    return scale_factor_clipped
