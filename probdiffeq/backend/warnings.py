"""Warnings."""

import warnings


def warn(msg, /, stacklevel):
    return warnings.warn(msg, stacklevel=stacklevel)


def filterwarnings(action, /) -> None:
    warnings.filterwarnings(action)
