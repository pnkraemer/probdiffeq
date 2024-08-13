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

from probdiffeq.backend import linalg, tree_util
from probdiffeq.backend import numpy as np


def revert_conditional_noisefree(R_X_F, R_X):
    """Like revert_conditional, but without observation noise."""
    if not R_X_F.shape[1] <= R_X_F.shape[0]:
        msg = (
            "Reverting noise-free conditionals requires "
            "that the conditional dimension is at most as "
            "large as the prior dimension. "
            f"Received: {R_X_F.shape[1]} >= {R_X_F.shape[0]}"
        )
        raise ValueError(msg)

    r_marg = triu_via_qr(R_X_F)
    crosscov = R_X.T @ R_X_F
    gain = linalg.cholesky_solve(r_marg.T, crosscov.T).T
    r_cor = R_X - R_X_F @ gain.T

    # TODO: only with this line is the output equivalent to the other function
    #  I don't like the double-QR decomposition --
    #  it feels that we don't save any computation here...
    if r_cor.shape[0] != r_cor.shape[1]:
        r_cor = triu_via_qr(r_cor)
    return r_marg, (r_cor, gain)


def revert_conditional(R_X_F, R_X, R_YX):
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
    if not _is_matrix(R_X) or not _is_matrix(R_YX) or not _is_matrix(R_X_F):
        msg = (
            "Unexpected tensor-dimension of the inputs."
            "\n\nExpected:\n\tR_X.shape=(n, n), "
            "R_X_F.shape=(n, k), R_YX.shape=(k, k)."
            f"\nReceived:\n\tR_X.shape={R_X.shape}, "
            f"R_X_F.shape={R_X_F.shape}, R_YX.shape={R_YX.shape}."
        )
        raise ValueError(msg)

    R = np.block([[R_YX, np.zeros((R_YX.shape[0], R_X.shape[1]))], [R_X_F, R_X]])
    R = triu_via_qr(R)

    # ~R_{Y}
    d_out = R_YX.shape[1]
    R_Y = R[:d_out, :d_out]

    # something like the cross-covariance
    R12 = R[:d_out, d_out:]

    # Implements G = R12.T @ np.linalg.inv(R_Y.T) in clever:
    G = linalg.solve_triangular(R_Y, R12, lower=False).T

    # ~R_{X \mid Y}
    R_XY = R[d_out:, d_out:]
    return R_Y, (R_XY, G)


def _is_matrix(mat, matrix_ndim=2):
    return np.ndim(mat) == matrix_ndim


def sum_of_sqrtm_factors(R_stack: tuple):
    r"""Compute the square root $R^\top R = R_1^\top R_1 + R_2^\top R_2 + ...$."""
    R = np.concatenate(R_stack)
    uppertri = triu_via_qr(R)
    if np.ndim(R_stack[0]) == 0:
        return np.reshape(uppertri, ())
    return uppertri


# logsumexp but for squares
def sqrt_sum_square_scalar(*args):
    """Compute sqrt(a**2 + b**2) without squaring a or b."""
    args_are_scalar = tree_util.tree_map(lambda x: np.ndim(x) == 0, args)
    if not tree_util.tree_all(args_are_scalar):
        args_shapes = tree_util.tree_map(np.shape, args)
        msg1 = "'sqrt_sum_square_scalar' expects scalar arguments. "
        msg2 = f"PyTree with shapes {args_shapes} received."
        raise ValueError(msg1 + msg2)

    stack = np.stack(args)
    sqrt_mat = triu_via_qr(stack[:, None])
    sqrt_mat_abs = np.abs(sqrt_mat)  # convention
    return np.reshape(sqrt_mat_abs, ())


def triu_via_qr(R, /):
    """Upper-triangularise a matrix using a QR-decomposition."""
    # TODO: enforce positive diagonals?
    #  (or expose this option; some equivalence tests might fail
    #   if we always use a positive diagonal.)
    return linalg.qr_r(R)
