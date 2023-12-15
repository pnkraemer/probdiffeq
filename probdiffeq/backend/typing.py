"""Typing module."""

import jax
from typing_extensions import TypeAlias  # typing.TypeAlias requires 3.10+

Array: TypeAlias = jax.Array
