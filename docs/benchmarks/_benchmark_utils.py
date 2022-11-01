"""Benchmark utils."""

import subprocess
import timeit

import jax
import numpy as np
from tqdm import tqdm
from tueplots import axes, bundles, cycler, markers

from odefilter import __version__ as odefilter_version


def plot_config():
    colors = ["cornflowerblue", "salmon", "mediumseagreen", "crimson", "darkorchid"]
    markers_ = ["o", "v", "P", "^", "X", "d"]
    return {
        **bundles.beamer_moml(rel_width=1.0, rel_height=1.0),
        **axes.color(base="black"),
        **axes.grid(),
        **cycler.cycler(
            marker=np.tile(markers_, 9)[:15],
            color=np.tile(colors, 10)[:15],
        ),
        **markers.with_edge(),
        "figure.dpi": 150,
    }


def print_info():

    commit = _most_recent_commit(abbrev=6)

    print(f"odefilter version:\n\t{odefilter_version}")
    print(f"Most recent commit:\n\t{commit}")
    print()
    jax.print_environment_info()


def _most_recent_commit(*, abbrev=21):
    return subprocess.check_output(
        ["git", "describe", "--always", f"--abbrev={abbrev}"]
    )


def workprecision(*, solve_fns, tols, **kwargs):
    results = {}
    for solve_fn, label in tqdm(solve_fns):

        times, errors = [], []

        for rtol in tols:

            def bench():
                return solve_fn(tol=rtol)

            t, error = time(bench, **kwargs)

            times.append(t)
            errors.append(error)

        results[label] = (times, errors)
    return results


def time(fn, /, *, number=5, repeat=5):
    res = fn()
    t = min(timeit.repeat(fn, number=number, repeat=repeat)) / number
    return t, res
