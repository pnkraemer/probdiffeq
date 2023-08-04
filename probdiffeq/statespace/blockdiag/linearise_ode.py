import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace.dense import linearise, variables


def constraint_0th(*, ode_order):
    def linearise_fun_wrapped(fun, mean):
        m0 = mean[:, :ode_order]
        fx = linearise.ts0(fun, m0.T)

        def a1(s):
            return s[:, ode_order, ...]

        return a1, -fx

    return linearise_fun_wrapped
