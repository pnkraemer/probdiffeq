"""Abstract base classes."""

import abc


class ABC(abc.ABC):  # noqa: B024,D101
    pass


def abstractmethod(*args):
    return abc.abstractmethod(*args)
