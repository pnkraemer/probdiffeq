"""Benchmark utils."""
import diffrax
import jax
import jax.experimental.ode
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from jax.typing import ArrayLike

from probdiffeq import ivpsolve


class FirstOrderIVP:
    """First-order IVPs in the ProbDiffEq API."""

    def __init__(self, vector_field, initial_values, t0, t1):
        """Construct a first-order IVP."""
        self.vector_field = vector_field
        self.initial_values = initial_values
        self.t0 = t0
        self.t1 = t1

    @property
    def args(self):
        return self.vector_field, self.initial_values, self.t0, self.t1

    @property
    def kwargs(self):
        return {}

    def to_jax(self, t):
        @jax.jit
        def func(u, t_, *p):
            return self.vector_field(u, t=t_, p=p)

        return JaxIVP(func, y0=self.initial_values[0], t=t)

    def to_scipy(self, t_eval):
        @jax.jit
        def fun(t, u, *p):
            return self.vector_field(u, t=t, p=p)

        return SciPyIVP(
            fun,
            t_span=(self.t0, self.t1),
            y0=np.asarray(self.initial_values[0]),
            t_eval=t_eval,
        )

    def to_diffrax(self):
        @jax.jit
        def f(t, y, args):
            return self.vector_field(y, t=t, p=args)

        return DiffraxIVP(
            diffrax.ODETerm(f),
            y0=self.initial_values[0],
            t0=self.t0,
            t1=self.t1,
        )


class DiffraxIVP:
    """Diffrax-implementation of IVPs."""

    def __init__(self, term, y0, t0, t1):
        """Construct a diffrax-implementation of an IVP."""
        self.term = term
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1

    @property
    def args(self):
        return (self.term,)

    @property
    def kwargs(self):
        return {
            "t0": self.t0,
            "t1": self.t1,
            "y0": self.y0,
            "dt0": None,
        }


class SecondOrderIVP:
    """First-order IVPs in the ProbDiffEq API."""

    def __init__(self, vector_field, initial_values, t0, t1):
        """Construct a second-order IVP."""
        self.vector_field = vector_field
        self.initial_values = initial_values
        self.t0 = t0
        self.t1 = t1

    @property
    def args(self):
        return self.vector_field, self.initial_values, self.t0, self.t1

    @property
    def kwargs(self):
        return {}


class JaxIVP:
    """JAX-implementation of an IVP."""

    def __init__(self, func, y0, t):
        """Construct a JAX-implementation of an IVP."""
        self.func = func
        self.y0 = y0
        self.t = t

    @property
    def args(self):
        return self.func, self.y0, self.t

    @property
    def kwargs(self):
        return {}


class SciPyIVP:
    """Scipy-implementation of an IVP."""

    def __init__(self, fun, t_span, y0, t_eval=None):
        """Construct a Scipy-implementation of an IVP."""
        self.fun = fun
        self.t_span = t_span
        self.y0 = y0
        self.t_eval = t_eval

    @property
    def args(self):
        return self.fun, self.t_span, self.y0

    @property
    def kwargs(self):
        return {"t_eval": self.t_eval}


def relative_rmse(*, solution: ArrayLike, atol=1e-5):
    """Relative root mean-squared error."""
    solution = jnp.asarray(solution)

    @jax.jit
    def error_fn(u: jax.Array, /):
        ratio = (u - solution) / (atol + solution)
        return jnp.linalg.norm(ratio) / jnp.sqrt(ratio.size)

    return error_fn


def absolute_rmse(*, solution: ArrayLike):
    """Relative root mean-squared error."""
    solution = jnp.asarray(solution)

    @jax.jit
    def error_fn(u: jax.Array, /):
        ratio = u - solution
        return jnp.linalg.norm(ratio) / jnp.sqrt(ratio.size)

    return error_fn


def probdiffeq_terminal_values(select_fn=None):
    def solve_fn(*args, atol, rtol, **kwargs):
        solution = ivpsolve.simulate_terminal_values(
            *args, atol=atol, rtol=rtol, **kwargs
        )
        if select_fn is not None:
            return select_fn(solution.u)
        return solution.u

    return solve_fn


def jax_terminal_values(select_fn=None):
    def solve_fn(*args, atol, rtol, **kwargs):
        solution = jax.experimental.ode.odeint(*args, atol=atol, rtol=rtol, **kwargs)
        if select_fn is not None:
            return select_fn(solution[-1, :])
        return solution[-1, :]

    return solve_fn


def scipy_terminal_values(select_fn=None):
    def solve_fn(*args, atol, rtol, **kwargs):
        solution = scipy.integrate.solve_ivp(*args, atol=atol, rtol=rtol, **kwargs)
        if select_fn is not None:
            return select_fn(solution.y.T[-1, :])
        return solution.y.T[-1, :]

    return solve_fn


def diffrax_terminal_values(select_fn=None):
    def solve_fn(*args, atol, rtol, **kwargs):
        controller = diffrax.PIDController(atol=atol, rtol=rtol)
        saveat = diffrax.SaveAt(t0=False, t1=True, ts=None)
        solution = diffrax.diffeqsolve(
            *args, saveat=saveat, stepsize_controller=controller, **kwargs
        )

        if select_fn is not None:
            return select_fn(solution.ys[0, :])
        return solution.ys[0, :]

    return solve_fn
