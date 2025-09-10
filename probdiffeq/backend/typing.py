"""Typing module."""

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar

import jax
from jax.typing import ArrayLike
from mypy_extensions import NamedArg

# Array
Array: TypeAlias = jax.Array
