"""Matrix-free API."""

from probdiffeq.backend import functools


def parametrised_linop(func, /, *, inputs, params=None):
    return functools.jacrev(lambda v: func(v, params))(inputs)
