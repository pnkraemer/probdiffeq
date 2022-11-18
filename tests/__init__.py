"""Tests."""

import warnings

from diffeqzoo import backend
from jax.config import config

# ODE examples must be in JAX
# todo: raise issue in diffeqzoo about this pylint-disable
backend.select("jax")  # pylint: disable=no-value-for-parameter

# All warnings shall be errors
warnings.filterwarnings("error")

# Test on CPU.
config.update("jax_platform_name", "cpu")

# Double precision
# Needed for equivalence tests for smoothers.
config.update("jax_enable_x64", True)
