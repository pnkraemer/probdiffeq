r"""Utilities for square root matrices and conditional distributions.

This function implements the efficient and robust
manipulation of square root matrices.

!!! warning "Matrix square root formats"

    The functions in this module assume **right** square root matrices,
    i.e., it assumes a factorisation $A=R^\top R$.
    The rest of the ``probdiffeq`` package assumes a **left** square
    root matrix, i.e., a factorisation $A = L L^\top$.

    _**Why?**_
    Because the functions in this module rely heavily on QR factorisations,
    and therefore the **right** square root matrix format is more natural.

    The rest of the ``probdiffeq`` package frequently maps
    $(m, L) \mapsto (Hm, HL)$, and therefore the **left** square root matrix
    is more natural.

"""
from typing import Tuple

import jax
import jax.numpy as jnp


def revert_conditional_noisefree(*, R_X_F, R_X):
    """Like revert_conditional, but without observation noise."""
    if not R_X_F.shape[1] <= R_X_F.shape[0]:
        msg = (
            "Reverting noise-free conditionals requires "
            "that the conditional dimension is at most as "
            "large as the prior dimension. "
            f"Received: {R_X_F.shape[1]} >= {R_X_F.shape[0]}"
        )
        raise ValueError(msg)

    r_marg = sqrtm_to_upper_triangular(R=R_X_F)
    crosscov = R_X.T @ R_X_F
    gain = jax.scipy.linalg.cho_solve((r_marg.T, True), crosscov.T).T
    r_cor = R_X - R_X_F @ gain.T

    # todo: only with this line is the output equivalent to the other function
    #  I don't like the double-QR decomposition --
    #  it feels that we don't save any computation here...
    if r_cor.shape[0] != r_cor.shape[1]:
        r_cor = sqrtm_to_upper_triangular(R=r_cor)
    return r_marg, (r_cor, gain)


# rename: reparametrise_conditional_correlation?
def revert_conditional(*, R_X_F, R_X, R_YX):
    r"""Revert the  square-root correlation in a Gaussian transition kernel.

    What does this mean? Assume we have two normally-distributed random variables,
    $X$ and $Y$.
    The joint distribution $p(X, Y)$ is Gaussian with some mean and some covariance.
    It can be parametrised by, among others, the conditionals

    $$
    p(X, Y) = p(X \mid Y) p(Y) = p(Y \mid X) p(X)
    $$

    Each marginal and each conditional are Gaussian, too.
    The means are easy to re-parametrise once the covariances are updated.

    _The present function provides the machinery to change the covariance
    parametrisation from $p(Y \mid X) p(X)$ to $p(X \mid Y) p(Y)$._

    !!! note "Application"

        This function is crucial to compute the backward transitions
        in Kalman-smoothing-like applications.


    The signature of this function is now

    $$
    (R_{Y}, (R_{X \mid Y}, G)) = \Phi(R_X, R_XF, R_{Y \mid X})
    $$

    and these quantities suffice to compute, e.g., smoothing posteriors
    and dense output. In the context of Kalman filtering,
    the matrix $G$ is often called the _Kalman gain_;
    in the context of Rauch-Tung-Striebel smoothing, it is called the
    _smoothing gain_.
    """
    if R_X.ndim != 2 or R_YX.ndim != 2 or R_X_F.ndim != 2:
        msg = (
            "Unexpected tensor-dimension of the inputs."
            "\n\nExpected:\n\tR_X.shape=(n, n), "
            "R_X_F.shape=(n, k), R_YX.shape=(k, k)."
            f"\nReceived:\n\tR_X.shape={R_X.shape}, "
            f"R_X_F.shape={R_X_F.shape}, R_YX.shape={R_YX.shape}."
        )
        raise ValueError(msg)

    R = jnp.block(
        [
            [R_YX, jnp.zeros((R_YX.shape[0], R_X.shape[1]))],
            [R_X_F, R_X],
        ]
    )
    # todo: point to sqrtm_to_upper_triangular()
    R = jnp.linalg.qr(R, mode="r")

    # ~R_{Y}
    d_out = R_YX.shape[1]
    R_Y = R[:d_out, :d_out]

    # something like the cross-covariance
    R12 = R[:d_out, d_out:]

    # Implements G = R12.T @ jnp.linalg.inv(R_Y.T) in clever:
    G = jax.scipy.linalg.solve_triangular(R_Y, R12, lower=False).T

    # ~R_{X \mid Y}
    R_XY = R[d_out:, d_out:]
    return R_Y, (R_XY, G)


def sum_of_sqrtm_factors(*, R_stack: Tuple):
    r"""Compute the square root $R^\top R = R_1^\top R_1 + R_2^\top R_2 + ...$."""
    R = jnp.vstack(R_stack)
    uppertri = sqrtm_to_upper_triangular(R=R)
    if jnp.ndim(R_stack[0]) == 0:
        return jnp.reshape(uppertri, ())
    return uppertri


def sqrtm_to_upper_triangular(*, R):
    """Transform a right matrix square root to a Cholesky factor."""
    # todo: enforce positive diagonals?
    upper_sqrtm = jnp.linalg.qr(R, mode="r")
    return upper_sqrtm
