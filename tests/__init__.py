"""Tests."""

import warnings

from diffeqzoo import backend
from jax.config import config

# ODE examples must be in JAX
backend.select("jax")

# All warnings shall be errors
warnings.filterwarnings("error")

# Test on CPU.
config.update("jax_platform_name", "cpu")
