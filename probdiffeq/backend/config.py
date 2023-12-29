"""Configuration management."""

import jax.config


def update(str_without_jax, value, /):
    jax.config.update(f"jax_{str_without_jax}", value)
