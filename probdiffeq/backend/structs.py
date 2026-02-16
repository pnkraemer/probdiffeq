"""Container types."""

import dataclasses
from enum import Enum  # noqa: F401
from typing import NamedTuple  # noqa: F401

from typing_extensions import dataclass_transform  # new in Python 3.11


@dataclass_transform()
def dataclass(*args, **kwargs):
    return dataclasses.dataclass(*args, frozen=True, **kwargs)
