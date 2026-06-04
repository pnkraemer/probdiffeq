"""Constraint functions and (posterior) linearization strategies.

Examples
--------
>>> from probdiffeq import probdiffeq

Construct ODE constraints as such:

>>> @probdiffeq.ode
... def vf(u, /, *, t):
...     return -u
>>>
>>> ssm = probdiffeq.state_space_model("dense")
>>> constraint = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
>>> print(constraint)
DenseOdeTs1(ode=ODEFunction(num_derivatives_in_args=1, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10)))


Implement high-order ODEs by passing a vector field with additional arguments as such:

>>> @probdiffeq.ode_second_order
... def vf(u, du, /, *, t):
...     return -du
>>>
>>> ssm = probdiffeq.state_space_model("isotropic")
>>> constraint = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
>>> print(constraint)
IsotropicOdeTs0(ode=ODEFunction(num_derivatives_in_args=2, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10)))


Or, use the constraint as a decorator

>>> import functools
>>>
>>> @functools.partial(probdiffeq.constraint_ode_ts0, ssm=ssm)
... @probdiffeq.ode_second_order
... def ode(u, du, /, *, t):
...     return -du
>>>
>>> print(ode)
IsotropicOdeTs0(ode=ODEFunction(num_derivatives_in_args=2, jacobian=jacobian_hutchinson_fwd(seed=1, num_probes=10)))


"""

from probdiffeq import ssm_impl
from probdiffeq._probdiffeq import problem_types
from probdiffeq.backend import flow, func, linalg, np, structs, tree
from probdiffeq.backend.typing import Array, Callable, Protocol, Sequence, TypeVar

__all__ = [
    "Constraint",
    "Linearization",
    "WeightedLeastSquaresNonlinearlyConstrained",
    "constraint_ode_ts0",
    "constraint_ode_ts1",
    "constraint_residual",
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

    $$
    x^\star = \arg\min_x \| L^{-1}(x - m)\|^2 ~~~\text{s.t.}~~~\text{constraint}(x) = 0,
    $$

    where $L$ is the Cholesky factor of a covariance matrix and $m$ a mean.
    """

    def __call__(self, constraint, x0, mean, cholesky):
        raise NotImplementedError


class wlstsq_nc_gauss_newton(WeightedLeastSquaresNonlinearlyConstrained):
    """Solve the weighted lstsq problem with nonlinear constraints using Gauss--Newton."""

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

    def __init__(
        self, wlstsq_nc: WeightedLeastSquaresNonlinearlyConstrained | None = None
    ) -> None:
        if wlstsq_nc is None:
            wlstsq_nc = wlstsq_nc_gauss_newton()
        self.wlstsq_nc = wlstsq_nc

    def __call__(self, constraint_flat: Callable, rv) -> Array:
        mean = rv.mean_flat
        mean, _info = self.wlstsq_nc(
            constraint_flat, mean, rv.mean_flat, rv.cholesky_flat
        )
        return mean


C = TypeVar("C", bound=Sequence)
"""A type-variable to describe sequences.

Used to type Taylor coefficients, for example.
"""


class Constraint(Protocol):
    """An interface for constraints + linearization in probabilistic solvers.

    Related:
    [`constraint_ode_ts0`](#probdiffeq.probdiffeq.constraint_ode_ts0),
    [`constraint_ode_ts1`](#probdiffeq.probdiffeq.constraint_ode_ts1),
    """

    init_linearization: Callable
    """Initialize the linearization of the constraint."""

    linearize: Callable
    """Linearize the constraint."""

    residual_order: int
    """The order of the root-constraint.

    Here, 'order' relates to the highest derivative that the
    constraint depends on; for instance, in first-order ODEs,
    the residual_order would be two; and in second-order ODEs,
    the residual_order would be three.
    """


def constraint_ode_ts0(
    ode: problem_types.ODEFunction, /, *, ssm: ssm_impl.FactSsmImpl
) -> Constraint:
    r"""Create an ODE constraint with zeroth-order Taylor linearisation.

    This constraint handles ODEs of the form

    $$
    \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), \frac{d^2u}{dt^2}(t), ..., t\right)
    $$

    where $k$ is the order of the ODE, which is read off the number of positional arguments in the vector field $f$ (argument `vf`).


    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    if not isinstance(ode, problem_types.ODEFunction):
        raise TypeError(ode)
    return ssm.linearize.ode_taylor_0th(ode=ode)


def constraint_ode_ts1(ode: problem_types.ODEFunction, /, *, ssm: ssm_impl.FactSsmImpl):
    r"""Create an ODE constraint and linearise with a first-order Taylor approximation.

    This constraint handles ODEs of the form

    $$
    \frac{d^k}{dt^k} u(t) = f\left(u(t), \frac{du}{dt}(t), \frac{d^2u}{dt^2}(t), ..., t\right)
    $$

    where $k$ is the order of the ODE.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    """
    if not isinstance(ode, problem_types.ODEFunction):
        raise TypeError(ode)
    return ssm.linearize.ode_taylor_1st(ode=ode)


def constraint_residual(
    residual: problem_types.Residual,
    *,
    ssm: ssm_impl.FactSsmImpl,
    linearization: Linearization | None = None,
):
    r"""Construct a general constraint.

    This constraint handles problems of the form

    $$
    f\left(u(t), \frac{du}{dt}(t), ..., \frac{d^k u}{dt^k}(t), t\right) = 0
    $$

    where $k$ is the order of the problem.


    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.


    Parameters
    ----------
    residual
        The residual to apply linearization to.
    ssm
        The state-space model to use for the constraint.
    linearization
        The strategy to use for finding the linearization point. If None, the prior mean is used as the linearization point.
        Adjust this variable to use posterior linearization (also known as iterated filtering).

    """
    if not isinstance(residual, problem_types.Residual):
        raise TypeError(residual)

    if linearization is None:
        linearization = linearization_prior_mean()
    return ssm.linearize.residual(residual=residual, linearization=linearization)


# def constraint_dae(
#     dae: problem_types.residual_from_stack,
#     *,
#     ssm: ssm_impl.FactSsmImpl,
#     linearization: Linearization | None = None,
# ):
#     r"""Like `constraint`, but for DAEs.

#     The advantage of a dedicated DAE constraint is that algebraic and differential
#     roots can enjoy different jet-orders, which increases accuracy.

#     This constraint handles problems of the form

#     $$
#     f\left(u(t), \frac{du}{dt}(t), ..., \frac{d^k u}{dt^k}(t), t\right) = 0,
#     ~~~
#     g\left(u(t), \frac{du}{dt}(t), ..., \frac{d^{k-1} u}{dt^{k-1}}(t), t\right) = 0
#     $$

#     where $f$ is the differential part and $g$ the algebraic part of the DAE. The order of the problem is read off the number of positional arguments in the algebraic root $g$ (argument `algebraic`), plus one. For instance, if the algebraic root has two positional arguments, then the problem is a second-order DAE. The differential root (argument `differential`) is expected to have one positional argument more than the algebraic root; in the previous example, the differential root would be expected to have three positional arguments.


#     !!! warning "Warning: highly EXPERIMENTAL feature!"
#         This function is highly experimental and not safe to use.
#         There is no guarantee that it works correctly (or at all).
#         It might be deleted tomorrow and without any deprecation policy.

#     Parameters
#     ----------
#     dae
#         The differential-algebraic equation system.
#     ssm
#         The state-space model to use for the constraint.
#     linearization
#         The strategy to use for finding the linearization point. If None, the prior mean is used as the linearization point.
#     """
#     if linearization is None:
#         linearization = linearization_prior_mean()

#     return ssm.linearize.dae(dae=dae, linearization=linearization)
