"""Typing module."""

from typing import Any, Callable, Generic, Optional, TypeVar  # noqa: F401

import jax
from typing_extensions import TypeAlias  # typing.TypeAlias requires 3.10+

# Array
Array: TypeAlias = jax.Array
