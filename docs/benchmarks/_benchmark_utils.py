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


def workprecision_make(*, solve_fns, tols, **kwargs):
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


def time(fn, /, *, number=3, repeat=5):
    res = fn()
    t = min(timeit.repeat(fn, number=number, repeat=repeat)) / number
    return t, res


def workprecision_plot(
    *,
    results,
    fig,
    ax,
    alpha=0.95,
    xticks=None,
    yticks=None,
    title="Work-precision diagram",
    ode_name=None,
    xlabel="Precision",
    xlabel_unit="RMSE, absolute",
    ylabel="Work",
    ylabel_unit="wall time, s",
    which_grid="both",
):
    for solver in results:
        times, errors = results[solver]
        ax.loglog(errors, times, label=solver, alpha=alpha)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_xticks(yticks)

    if title is not None:
        if ode_name is not None:
            title += f" [{ode_name}]"
        ax.set_title(title)
    if xlabel is not None:
        if ylabel_unit is not None:
            xlabel += f" [{xlabel_unit}]"
            ax.set_xlabel(xlabel)
    if ylabel is not None:
        if ylabel_unit is not None:
            ylabel += f" [{ylabel_unit}]"
            ax.set_ylabel(ylabel)

    ax.grid(which_grid)
    ax.legend()

    return fig, ax
