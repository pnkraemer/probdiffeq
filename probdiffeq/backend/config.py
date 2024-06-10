"""Configuration management."""

import jax


def update(str_without_jax, value, /):
    jax.config.update(f"jax_{str_without_jax}", value)
