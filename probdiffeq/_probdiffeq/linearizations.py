from probdiffeq import ssm_impl
from probdiffeq.backend import flow, func, linalg, np, structs, tree
from probdiffeq.backend.typing import Array, Callable, TypeVar

__all__ = [
    "Linearization",
    "WeightedLeastSquaresNonlinearlyConstrained",
    "linearization_map",
    "linearization_prior_mean",
    "wlstsq_nc_gauss_newton",
]

N = TypeVar("N", bound=ssm_impl.AbstractTreeNormal)
"""A type-variable to describe normal distributions.

Used to type marginals, for example.
"""


class Linearization:
    """Find a linearization point.

    Note how this object handles *where* to linearize, but not *how* to linearize.
    """

    def __call__(self, constraint_flat: Callable, rv: N) -> Array:
        """Find a linearization point.

        Parameters
        ----------
        constraint_flat
            The constraint to linearize, flattened to work with the raveled mean.
        rv
            The distribution to linearize around. The mean of this distribution is typically used as the linearization point.

        Returns
        -------
        Array
            The point at which to linearize the constraint.
        """
        raise NotImplementedError


class WeightedLeastSquaresNonlinearlyConstrained:
    r"""Solve nonlinearly constrained least-squares problems.

    Concretely, solve problems of the form

    \min_x \| L^{-1}(x - mean)\|^2 s.t. constraint(x) = 0,

    where L is the Cholesky factor of a covariance matrix.
    """

    def __call__(self, constraint, x0, mean, cholesky):
        raise NotImplementedError


class wlstsq_nc_gauss_newton(WeightedLeastSquaresNonlinearlyConstrained):
    """Solve the weighted lstsq problem with nonlinear constraints using Gauss--Newton."""

    def __init__(
        self, *, maxiter, tol, lstsq=linalg.lstsq_svd, while_loop=flow.while_loop
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.lstsq = lstsq
        self.while_loop = while_loop

    def __call__(self, constraint, x0, mean, cholesky):
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
            cond1 = linalg.vector_norm(state.fx) > self.tol * np.sqrt(state.fx.size)

            # 2. Maxiter not yet reached
            cond2 = state.i < self.maxiter

            # 3. Iterations not yet converged
            cond3 = linalg.vector_norm(state.dx) > self.tol * np.sqrt(state.dx.size)

            # If any of the conditions is violated, the iteration is over.
            return np.logical_and(np.logical_and(cond1, cond2), cond3)

        def body_fun(state: State) -> State:
            Jx = func.jacfwd(constraint)(state.x)

            H = Jx @ cholesky
            r = state.fx + Jx @ (mean - state.x)
            dy = self.lstsq(H, r)
            dx = mean - state.x - cholesky @ dy
            xnew = state.x + dx

            fxnew = constraint(xnew)
            return State(xnew, fxnew, dx=xnew - state.x, i=state.i + 1)

        init = State(x0, constraint(x0), dx=np.ones_like(x0), i=0)
        final = self.while_loop(cond_fun, body_fun, init=init)
        return final.x, {"iters": final.i}


class linearization_prior_mean(Linearization):
    """Linearization point is the prior mean."""

    def __call__(self, constraint_flat: Callable, rv) -> Array:
        del constraint_flat
        return rv.mean_flat


class linearization_map(Linearization):
    """Linearization point is the maximum-a-posteriori estimate."""

    def __init__(self, wlstsq_nc: WeightedLeastSquaresNonlinearlyConstrained) -> None:
        self.wlstsq_nc = wlstsq_nc

    def linearization_point(self, constraint_flat: Callable, rv) -> Array:
        mean = rv.mean_flat
        mean, _info = self.wlstsq_nc(
            constraint_flat, mean, rv.mean_flat, rv.cholesky_flat
        )
        return mean
