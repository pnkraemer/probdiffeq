"""Timing utilities."""

import time
import timeit


def perf_counter():
    return time.perf_counter()


def repeat(fun, /, *, repeats):
    return list(timeit.repeat(fun, number=1, repeat=repeats))
