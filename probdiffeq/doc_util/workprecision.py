"""Work-precision diagram utils."""

import statistics
import timeit
from typing import Any, Callable, Dict, NamedTuple

import jax
from tqdm.auto import tqdm as progressbar

from probdiffeq import _control_flow


class MethodConfig(NamedTuple):
    """Work-precision diagram configuration for a single method."""

    method: Dict
    label: str


class ProblemConfig(NamedTuple):
    """Work-precision diagram configuration for a given problem."""

    label: str
    problem: Any
    solve_fn: Callable
    error_fn: Callable


class Results(NamedTuple):
    """Work-precision diagram results."""

    work: float
    precision: float


class Timings(NamedTuple):
    """Work-precision diagram timings."""

    min: float
    max: float
    mean: float
    stdev: float

    @classmethod
    def from_list(cls, timings):
        return cls(
            min=min(timings),
            max=max(timings),
            mean=statistics.mean(timings),
            stdev=statistics.stdev(timings),
        )


# Work-precision diagram functionality


def create(problem, methods, tolerances, *, repeat=5):
    return {
        method.label: _control_flow.tree_stack(
            [
                _evaluate_method(
                    method_config=method,
                    problem_config=problem,
                    tolerance=tol,
                    repeat=repeat,
                )
                for tol in progressbar(tolerances, leave=False)
            ]
        )
        for method in progressbar(methods)
    }


def _evaluate_method(*, method_config, problem_config, tolerance, repeat):
    problem = problem_config.problem
    error_fn = problem_config.error_fn
    solve_fn = problem_config.solve_fn
    method = method_config.method

    @jax.jit
    def fn():
        return solve_fn(*problem, tol=tolerance, **method)

    qoi = fn()
    error = error_fn(qoi)
    timings = timeit.repeat(fn, number=1, repeat=repeat)
    return Results(work=Timings.from_list(timings), precision=error)


def plot(
    *,
    results: Results,
    fig,
    ax,
    title,
    alpha=0.95,
    xticks=None,
    yticks=None,
    xlabel="Precision",
    xlabel_unit="RMSE, relative",
    ylabel="Work",
    ylabel_unit="wall time, s",
    which_grid="both",
):
    for solver, result in results.items():
        ax.loglog(result.precision, result.work.mean, label=solver, alpha=alpha)
        ax.fill_between(
            result.precision,
            result.work.mean - result.work.stdev,
            result.work.mean + result.work.stdev,
            alpha=0.25 * alpha,
        )

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_xticks(yticks)

    if title is not None:
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
    print(
        "Next up: compute WP for all tolerances at once, "
        "upgrade the remaining benchmarks. "
        "Figure out how to squeeze other solvers into this framework."
    )
    return fig, ax
