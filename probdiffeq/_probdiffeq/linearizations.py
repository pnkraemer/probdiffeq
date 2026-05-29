"""Posterior linearization API."""

from probdiffeq.backend import flow, func, linalg, np, structs, tree
from probdiffeq.backend.typing import Array, Callable

__all__ = [
    "linearization_map",
    "linearization_prior_mean",
    "nlstsq_gauss_newton_weighted_constrained",
]


def linearization_prior_mean() -> Callable:
    """Linearization point is the prior mean."""

    def linearization_point(constraint_flat: Callable, rv) -> Array:
        del constraint_flat
        return rv.mean_flat

    return linearization_point


def linearization_map(nlstsq: Callable) -> Callable:
    """Linearization point is the posterior mean."""

    def linearization_point(constraint_flat: Callable, rv) -> Array:
        mean = rv.mean_flat
        mean, _info = nlstsq(constraint_flat, mean, rv.mean_flat, rv.cholesky_flat)
        return mean

    return linearization_point


def nlstsq_gauss_newton_weighted_constrained(
    *, maxiter, tol, lstsq=linalg.lstsq_svd, while_loop=flow.while_loop
):
    r"""Solve nonlinearly constrained least-squares problems.

    Concretely, solve problems of the form

    \min_x \| L^{-1}(x - mean)\|^2 s.t. constraint(x) = 0,

    where L is the Cholesky factor of a covariance matrix.
    """

    def solve(constraint, x0, mean, cholesky):
        @tree.register_dataclass
        @structs.dataclass
        class State:
            x: Array
            fx: Array
            dx: Array
            i: int

        def cond_fun(state: State):
            # Three conditions that all need to be satisfied to continue:

            # 1. Constraint not yet satisfied
            cond1 = linalg.vector_norm(state.fx) > tol * np.sqrt(state.fx.size)

            # 2. Maxiter not yet reached
            cond2 = state.i < maxiter

            # 3. Iterations not yet converged
            cond3 = linalg.vector_norm(state.dx) > tol * np.sqrt(state.dx.size)

            # If any of the conditions is violated, the iteration is over.
            return np.logical_and(np.logical_and(cond1, cond2), cond3)

        def body_fun(state: State) -> State:
            Jx = func.jacfwd(constraint)(state.x)

            H = Jx @ cholesky
            r = state.fx + Jx @ (mean - state.x)
            dy = lstsq(H, r)
            dx = mean - state.x - cholesky @ dy
            xnew = state.x + dx

            fxnew = constraint(xnew)
            return State(xnew, fxnew, dx=xnew - state.x, i=state.i + 1)

        init = State(x0, constraint(x0), dx=np.ones_like(x0), i=0)
        final = while_loop(cond_fun, body_fun, init=init)
        return final.x, {"iters": final.i}

    return solve
