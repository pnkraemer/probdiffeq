from probdiffeq.backend import linalg, np, tree
from probdiffeq.backend.typing import Sequence

__all__ = ["dt0", "dt0_adaptive"]


def dt0(vf, initial_values: Sequence, /, scale=0.01, nugget=1e-5, **vf_kwargs):
    """Propose an initial time-step."""
    f0 = vf(*initial_values, **vf_kwargs)

    u0, *_ = initial_values
    u0, _ = tree.ravel_pytree(u0)
    f0, _ = tree.ravel_pytree(f0)

    norm_y0 = linalg.vector_norm(u0)
    norm_dy0 = linalg.vector_norm(f0) + nugget

    return scale * norm_y0 / norm_dy0


def dt0_adaptive(
    vf, initial_values: Sequence, /, t0, *, error_contraction_rate, rtol, atol
):
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

    f0 = vf(*initial_values, t=t0)

    y0, unravel = tree.ravel_pytree(y0)
    f0, _ = tree.ravel_pytree(f0)

    scale = atol + np.abs(y0) * rtol
    d0, d1 = linalg.vector_norm(y0), linalg.vector_norm(f0)

    dt0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + dt0 * f0
    f1 = vf(unravel(y1), t=t0 + dt0)
    f1, _ = tree.ravel_pytree(f1)
    d2 = linalg.vector_norm((f1 - f0) / scale) / dt0

    dt1 = np.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        np.maximum(1e-6, dt0 * 1e-3),
        (0.01 / np.maximum(d1, d2)) ** (1.0 / (error_contraction_rate + 1.0)),
    )
    return np.minimum(100.0 * dt0, dt1)
