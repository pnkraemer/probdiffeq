r"""Utilities for square root matrices and conditional distributions.

!!! warning "Matrix square root formats"

    The functions in this module assume **right** square root matrices,
    i.e., it assumes a factorisation $A=R^\top R$.
    The rest of the ``odefilter`` package assumes a **left** square
    root matrix, i.e., a factorisation $A = L L^\top$.

    _**Why?**_
    Because the functions in this module rely heavily on QR factorisations,
    and therefore the **right** square root matrix format is more natural.

    The rest of the ``odefilter`` package frequently maps
    $(m, L) \mapsto (Hm, HL)$, and therefore the **left** square root matrix
    is more natural.

"""

import jax.numpy as jnp
import jax.scipy as jsp


# todo: clean up and make R- instead of L-based.
def revert_gauss_markov_correlation(*, R_X_F, R_X, R_YX):
    r"""Revert the correlation structure of a Gaussian transition kernel \
    without leaving the square-root parametrisation.

    What does this mean? Assume we have two normally-distributed random variables,
    $X$ and $Y$.
    The joint distribution $p(X, Y)$ is Gaussian with some mean and some covariance.
    It can be parametrised by, among others, the conditionals.

    $$
    p(X, Y) = p(X \mid Y) p(Y) = p(Y \mid X) p(X)
    $$

    Each marginal and each conditional are Gaussian, too.
    The means are easy to reparametrise once the covariances are updated.

    _The present function provides the machinery to change the covariance
    parametrisation from $p(Y \mid X) p(X)$ to $p(X \mid Y) p(Y)$._
    Let $p(X) = N(..., R_X^\top R_X)$ and $p(Y \mid X) = N(F^\top X, R_{Y \mid X}^\top R_{Y \mid X})$.
    Then, the joint covariance of $X$ and $Y$ is

    $$
    \mathrm{cov}(X, Y) =
    \begin{pmatrix}
        R_X^\top R_X & R_X^\top (R_X F) \\
        (R_X F)^\top R_X & (R_X F)^\top (R_X F) + R_{Y \mid X}^\top R_{Y \mid X}
    \end{pmatrix}
    $$

    The marginal

    $$
    p(Y) = N(..., R_{Y}^\top R_{Y}) =
    N(..., (R_X R_F)^\top R_X R_F + R_{Y \mid X}^\top R_{Y \mid X})
    $$

    could also be the starting point of a different parametrisation;
    let $p(X \mid Y) = N(G^\top Y, R_{X \mid Y}^\top R_{X \mid Y})$, then the joint covariance
    of $X$ and $Y$ is

    $$
    \mathrm{cov}(X, Y) =
    \begin{pmatrix}
        (R_{Y} G)^\top (R_{Y} G) + R_{X \mid Y}^\top R_{X \mid Y} & R_{Y}^\top (R_{Y} G) \\
        (R_{Y} G)^\top R_{Y} &  R_{Y}^\top R_{Y}
    \end{pmatrix}
    $$

    This "change of direction" is useful to compute the smoothing gains
    of a Rauch-Tung-Striebel smoother during the forward-filtering pass.

    !!! note "Application"

        This function is absolutely crucial to compute the backward transitions
        in Kalman-smoothing-like applications.

    The signature of this function is now

    $$
    (R_{Y}, (R_{X \mid Y}, G)) = \Phi(R_X, R_XF, R_{Y \mid X})
    $$

    and these quantities suffice to compute, e.g., smoothing posteriors
    and dense output. In the context of Kalman filtering,
    the matrix $G$ is often called the _Kalman gain_;
    in the context of Rauch-Tung-Striebel smoothing, it is called hte
    _smoothing gain_.


    """

    R = jnp.block(
        [
            [R_YX, jnp.zeros_like(R_X_F.T)],
            [R_X_F, R_X],
        ]
    )
    R = jnp.linalg.qr(R, mode="r")
    d = R_YX.shape[0]

    # ~R_{Y}
    R1 = R[:d, :d]
    R_Y = _make_diagonal_positive(R=R1)

    # something like the cross-covariance
    R12 = R[:d, d:]
    G = jsp.linalg.solve_triangular(R_Y, R12).T

    # ~R_{X \mid Y}
    R3 = R[d:, d:]
    R_XY = _make_diagonal_positive(R=R3)
    return R_Y, (R_XY, G)


def sum_of_sqrtm_factors(*, R1, R2):
    """Compute Cholesky factor of R1.T @ R1 + R2.T @ R2."""
    R = jnp.vstack((R1, R2))
    chol = sqrtm_to_cholesky(R=R)
    if jnp.ndim(R1) == 0:
        return jnp.reshape(chol, ())
    return chol


def sqrtm_to_cholesky(*, R):
    """Transform a matrix square root to a Cholesky factor."""
    upper_sqrtm = jnp.linalg.qr(R, mode="r")
    return _make_diagonal_positive(R=upper_sqrtm)


def _make_diagonal_positive(*, R):
    s = jnp.sign(jnp.diag(R))
    x = jnp.where(s == 0, 1, s)
    return x[..., None] * R
