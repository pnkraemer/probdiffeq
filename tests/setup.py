"""Test-setup."""

import warnings

import jax.config

from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode
from probdiffeq.impl import impl

# All warnings shall be errors
warnings.filterwarnings("error")

# Test on CPU.
jax.config.update("jax_platform_name", "cpu")

# Double precision
# Needed for equivalence tests for smoothers.
jax.config.update("jax_enable_x64", True)


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
            return ode.ivp_logistic()
        return ode.ivp_lotka_volterra()

    def ode_affine(self):
        if self._which == "scalar":
            return ode.ivp_affine_scalar()
        return ode.ivp_affine_multi_dimensional()

    def rv(self):
        output_scale = np.ones_like(impl.prototypes.output_scale())
        discretise_func = impl.ssm_util.ibm_transitions(3, output_scale)
        (_matrix, rv), _pre = discretise_func(0.5)
        return rv


setup = _Setup()
