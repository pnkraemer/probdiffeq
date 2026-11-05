"""Container types."""

import dataclasses
from enum import Enum  # noqa: F401
from typing import (
    NamedTuple,  # noqa: F401
    dataclass_transform,  # new in Python 3.11
)

from jax import ShapeDtypeStruct  # noqa: F401


@dataclass_transform()
def dataclass(*args, **kwargs):
    return dataclasses.dataclass(*args, frozen=True, **kwargs)


def dataclass_field(metadata):
    return dataclasses.field(metadata=metadata)
