"""Typing module."""

from collections.abc import Callable, Sequence
from typing import Any, Generic, Optional, TypeAlias, TypeVar

import jax
from mypy_extensions import NamedArg

# Array
Array: TypeAlias = jax.Array
