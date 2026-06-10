"""Linearization-point strategies for probabilistic solvers.

Examples
--------
>>> from probdiffeq import probdiffeq

Use the prior mean as the linearization point (default):

>>> lp = probdiffeq.linearization_point_prior()
>>> print(lp)
linearization_point_prior()

Use the MAP estimate as the linearization point for iterated filtering:

>>> lp = probdiffeq.linearization_point_maximum_a_posteriori()
>>> print(lp)
linearization_point_maximum_a_posteriori(nlstsq=lstsq_constrained_gauss_newton())

"""

from probdiffeq._probdiffeq.ssm_via_api import LinearizationPoint
from probdiffeq.backend import flow, func, linalg, np, structs, tree
from probdiffeq.backend.typing import Array, Callable

__all__ = [
    "LstSqConstrained",
    "linearization_point_maximum_a_posteriori",
    "linearization_point_prior",
    "lstsq_constrained_gauss_newton",
]


class LstSqConstrained:
    r"""Solve nonlinearly-constrained, weighted least-squares problems.

    Concretely, solve problems of the form

    $$
    x^\star = \arg\min_x \| L^{-1}(x - m)\|^2 ~~~\text{s.t.}~~~\text{constraint}(x) = 0,
    $$

    where $L$ is the Cholesky factor of a covariance matrix and $m$ a mean.
    """

    def __call__(self, constraint, x0, mean, cholesky):
        raise NotImplementedError


class lstsq_constrained_gauss_newton(LstSqConstrained):
    """Solve the constrained LstSq problem using Gauss--Newton."""

    def __init__(
        self,
        *,
        maxiter=10,
        tol=1e-6,
        lstsq=linalg.lstsq_svd,
        while_loop=flow.while_loop,
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.lstsq = lstsq
        self.while_loop = while_loop

    def __repr__(self) -> str:
        return "lstsq_constrained_gauss_newton()"

    def __call__(self, constraint, x0, mean, cholesky, **constraint_kwargs):

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
            Jx = func.jacfwd(lambda s: constraint(s, **constraint_kwargs))(state.x)
            H = Jx @ cholesky
            r = state.fx + Jx @ (mean - state.x)
            dy = self.lstsq(H, r)
            dx = mean - state.x - cholesky @ dy
            xnew = state.x + dx

            fxnew = constraint(xnew, **constraint_kwargs)
            return State(xnew, fxnew, dx=xnew - state.x, i=state.i + 1)

        init = State(x0, constraint(x0, **constraint_kwargs), dx=np.ones_like(x0), i=0)
        final = self.while_loop(cond_fun, body_fun, init=init)
        stats = {
            "iters": final.i,
            "final_constraint": final.fx,
            "final_increment": final.dx,
        }
        return final.x, stats


class linearization_point_prior(LinearizationPoint):
    """Linearization point is the prior mean."""

    def __call__(self, constraint_flat: Callable, rv, **constraint_kwargs) -> Array:
        del constraint_flat
        del constraint_kwargs
        return rv.mean_flat


class linearization_point_maximum_a_posteriori(LinearizationPoint):
    """Linearization point is the maximum-a-posteriori estimate."""

    def __init__(self, nlstsq: LstSqConstrained | None = None) -> None:
        if nlstsq is None:
            nlstsq = lstsq_constrained_gauss_newton()
        self.nlstsq = nlstsq

    def __repr__(self) -> str:
        return f"linearization_point_maximum_a_posteriori(nlstsq={self.nlstsq!r})"

    def __call__(self, constraint_flat: Callable, rv, **constraint_kwargs) -> Array:
        mean = rv.mean_flat
        mean, _info = self.nlstsq(
            constraint_flat, mean, rv.mean_flat, rv.cholesky_flat, **constraint_kwargs
        )
        return mean
