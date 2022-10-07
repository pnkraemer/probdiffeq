"""Tests."""

from diffeqzoo import backend
from jax.config import config

backend.select("jax")


config.update("jax_enable_x64", True)
