"""Work-precision diagram utils."""

import statistics
import timeit
from typing import Any, Callable, Dict, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tqdm.auto import tqdm

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
    atols: ArrayLike
    rtols: ArrayLike
    repeat: int = 5


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
    data: ArrayLike

    @classmethod
    def from_list(cls, timings):
        return cls(
            min=min(timings),
            max=max(timings),
            mean=statistics.mean(timings),
            stdev=statistics.stdev(timings),
            data=jnp.asarray(timings),
        )


# Work-precision diagram functionality


def create(problem, methods):
    return {
        method.label: _evaluate_method(method_config=method, problem_config=problem)
        for method in tqdm(methods)
    }


def _evaluate_method(*, method_config, problem_config):
    problem = problem_config.problem
    method = method_config.method

    error_fn = problem_config.error_fn
    solve_fn = problem_config.solve_fn
    atols = problem_config.atols
    rtols = problem_config.rtols
    repeat = problem_config.repeat

    @jax.jit
    def fn(atol, rtol):
        return solve_fn(*problem, atol=atol, rtol=rtol, **method)

    return _control_flow.tree_stack(
        [
            _evaluate_method_and_tolerance(
                error_fn=error_fn, fn=fn, rtol=rtol, atol=atol, repeat=repeat
            )
            for atol, rtol in zip(tqdm(atols, leave=False), rtols)
        ]
    )


def _evaluate_method_and_tolerance(*, error_fn, fn, atol, rtol, repeat):
    qoi = fn(atol=atol, rtol=rtol)
    error = error_fn(qoi)
    timings = timeit.repeat(lambda: fn(atol=atol, rtol=rtol), number=1, repeat=repeat)
    return Results(work=Timings.from_list(timings), precision=error)


def plot(
    *,
    results: Results,
    fig,
    ax,
    title,
    alpha_mean=0.95,
    alpha_stdev=0.2 * 0.95,
    xticks=None,
    yticks=None,
    xlabel="Precision",
    xlabel_unit="RMSE, relative",
    ylabel="Work",
    ylabel_unit="wall time, s",
    which_grid="both",
):
    for solver, result in results.items():
        ax.loglog(result.precision, result.work.mean, label=solver, alpha=alpha_mean)
        ax.fill_between(
            result.precision,
            result.work.mean - result.work.stdev,
            result.work.mean + result.work.stdev,
            alpha=alpha_stdev,
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
    return fig, ax
