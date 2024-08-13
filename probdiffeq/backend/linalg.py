"""Linear-algebra routines."""

import jax.numpy as jnp
import jax.scipy.linalg


@jax.custom_jvp
def qr_r(arr, /):
    return jnp.linalg.qr(arr, mode="r")


@qr_r.defjvp
def qr_r_jvp(primals, tangents):
    """Evaluate the JVP of qr_r.

    The difference to JAX's custom JVP for the QR-decomposition
    is that qr_r does not return Q which removes the linear solve
    with R from the computation.
    This is not only cheaper, but also more stable because it makes qr_r
    differentiable at the origin (which means calling it with the zero matrix).
    Using the JVP of the full QR decomposition does not have this feature.

    Refer to Issue #668 for why we need this.
    """
    # todo: maybe the QR decomposition should not be differentiable at the origin...
    #  but what we definitely want is that triangularisation (which calls qr_r) is
    #  differentiable at the origin. See #668.
    #  But for now, we don't distinguish between those two cases.
    (M,) = primals
    (M_dot,) = tangents
    Q, R = jnp.linalg.qr(M, mode="reduced")

    # Treat 'Q' as constant, which implies
    # R = Q^\top M and we get obvious derivatives
    R_dot = Q.T @ M_dot
    return R, R_dot


# All Cholesky factors are lower-triangular by default


def cholesky_factor(arr, /):
    return jnp.linalg.cholesky(arr)


# All Cholesky factors are lower-triangular by default


def cholesky_solve(arr, rhs, /):
    return jax.scipy.linalg.cho_solve((arr, True), rhs)


def vector_norm(arr, /, *, order=None):
    return jnp.linalg.norm(arr, ord=order)


def solve_triangular(matrix, rhs, /, *, trans=0, lower=False):
    return jax.scipy.linalg.solve_triangular(matrix, rhs, trans=trans, lower=lower)


def inv(matrix, /):
    return jnp.linalg.inv(matrix)


def vector_dot(a, b, /):
    return jnp.dot(a, b)


def diagonal_along_axis(arr, /, *, axis1, axis2):
    return jnp.diagonal(arr, axis1=axis1, axis2=axis2)


def diagonal(arr, /):
    return jnp.diagonal(arr)


def triu(arr, /):
    return jnp.triu(arr)
