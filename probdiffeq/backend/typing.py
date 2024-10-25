"""Typing module."""

from collections.abc import Sequence  # noqa: F401
from typing import Any, Callable, Generic, Optional, TypeVar  # noqa: F401

import jax
from mypy_extensions import NamedArg  # noqa: F401

# typing.TypeAlias requires 3.10+
from typing_extensions import TypeAlias

# Array
Array: TypeAlias = jax.Array
