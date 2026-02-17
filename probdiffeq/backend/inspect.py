"""Light wrapper around the 'inspect' module."""

import inspect


def signature(fun, /):
    return inspect.signature(fun)


Parameter = inspect.Parameter
