"""Benchmark utils."""
import jax
import jax.experimental.ode
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from jax.typing import ArrayLike

from probdiffeq import solution_routines


class FirstOrderIVP:
    """First-order IVPs in the ProbDiffEq API."""

    def __init__(self, vector_field, initial_values, t0, t1):
        self.vector_field = vector_field
        self.initial_values = initial_values
        self.t0 = t0
        self.t1 = t1

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

    @property
    def args(self):
        return self.vector_field, self.initial_values, self.t0, self.t1

    @property
    def kwargs(self):
        return {}


class JaxIVP:
    def __init__(self, func, y0, t):
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
    def __init__(self, fun, t_span, y0, t_eval=None):
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
    def error_fn(u: ArrayLike, /):
        ratio = (u - solution) / (atol + solution)
        return jnp.linalg.norm(ratio) / jnp.sqrt(ratio.size)

    return error_fn


def probdiffeq_terminal_values():
    def solve_fn(*args, atol, rtol, **kwargs):
        solution = solution_routines.simulate_terminal_values(
            *args, atol=atol, rtol=rtol, **kwargs
        )
        return solution.u

    return solve_fn


def jax_terminal_values():
    def solve_fn(*args, atol, rtol, **kwargs):
        solution = jax.experimental.ode.odeint(*args, atol=atol, rtol=rtol, **kwargs)
        return solution[-1, :]

    return solve_fn


def scipy_terminal_values():
    def solve_fn(*args, atol, rtol, **kwargs):
        solution = scipy.integrate.solve_ivp(*args, atol=atol, rtol=rtol, **kwargs)
        return solution.y.T[-1, :]

    return solve_fn
