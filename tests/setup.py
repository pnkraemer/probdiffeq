"""Test-setup."""

import warnings

import diffeqzoo
import diffeqzoo.ivps
import jax
import jax.numpy as jnp
from diffeqzoo import backend
from jax.config import config

from probdiffeq.impl import impl

# ODE examples must be in JAX
backend.select("jax")

# All warnings shall be errors
warnings.filterwarnings("error")

# Test on CPU.
config.update("jax_platform_name", "cpu")

# Double precision
# Needed for equivalence tests for smoothers.
config.update("jax_enable_x64", True)


class _Setup:
    def __init__(self):
        self._which = None

    def select(self, which, /):
        if which == "scalar":
            impl.select("scalar")
        else:
            impl.select(which, ode_shape=(2,))

        self._which = which

    def ode(self):
        if self._which == "scalar":
            return self._ode_scalar()

        return self._ode_multi_dimensional()

    @staticmethod
    def _ode_scalar():
        f, u0, (t0, _), f_args = diffeqzoo.ivps.logistic()
        t1 = 0.75

        @jax.jit
        def vf(x, *, t):  # noqa: ARG001
            return f(x, *f_args)

        return vf, (u0,), (t0, t1)

    @staticmethod
    def _ode_multi_dimensional():
        f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
        t1 = 2.0  # Short time-intervals are sufficient for this test.

        @jax.jit
        def vf(x, *, t):  # noqa: ARG001
            return f(x, *f_args)

        return vf, (u0,), (t0, t1)

    def ode_affine(self):
        if self._which == "scalar":
            return self._ode_affine_scalar()
        return self._ode_affine_multi_dimensional()

    @staticmethod
    def _ode_affine_multi_dimensional():
        t0, t1 = 0.0, 2.0
        u0 = jnp.ones((2,))

        @jax.jit
        def vf(x, *, t):  # noqa: ARG001
            return 2 * x

        def solution(t):
            return jnp.exp(2 * t) * jnp.ones((2,))

        return vf, (u0,), (t0, t1), solution

    @staticmethod
    def _ode_affine_scalar():
        t0, t1 = 0.0, 2.0
        u0 = 1.0

        @jax.jit
        def vf(x, *, t):  # noqa: ARG001
            return 2 * x

        def solution(t):
            return jnp.exp(2 * t)

        return vf, (u0,), (t0, t1), solution

    def rv(self):
        output_scale = jnp.ones_like(impl.prototypes.output_scale())
        discretise_func = impl.ssm_util.ibm_transitions(3, output_scale)
        (_matrix, rv), _pre = discretise_func(0.5)
        return rv


setup = _Setup()
