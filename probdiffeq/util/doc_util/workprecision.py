"""Work-precision diagram utils."""

import statistics
import timeit
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from probdiffeq.backend import tree_array_util


class MethodConfig:
    """Work-precision diagram configuration for a single method."""

    def __init__(
        self, method, label, key=None, jit=True, tols_static=False, plotting_kwargs=None
    ):
        """Construct a method-configuration."""
        self.method = method
        self.label = label
        self.key = key if key is not None else "probdiffeq"
        self.jit = jit
        self.tols_static = tols_static
        self.plotting_kwargs = plotting_kwargs if plotting_kwargs is not None else {}


class ProblemConfig:
    """Work-precision diagram configuration for a given problem."""

    def __init__(
        self,
        label,
        problems,
        solve_fns,
        error_fn,
        atols,
        rtols,
        error_unit="RMSE",
        repeat=5,
    ):
        """Construct a problem-configuration."""
        self.label = label

        # TODO: the below is suboptimal.
        #  Instead, we should save as is, and in the benchmark
        #  check whether something is a dict (in which case we look for keys) or not.
        if not isinstance(problems, dict):
            problems = {"probdiffeq": problems}
        self.problems = problems

        if not isinstance(solve_fns, dict):
            solve_fns = {"probdiffeq": solve_fns}
        self.solve_fns = solve_fns

        self.error_fn = error_fn
        self.error_unit = error_unit

        self.atols = atols
        self.rtols = rtols

        self.repeat = repeat


@jax.tree_util.register_pytree_node_class
class Results:
    """Work-precision diagram results."""

    def __init__(self, work, precision):
        """Construct a result-collection."""
        self.work = work
        self.precision = precision

    def tree_flatten(self):
        children = self.work, self.precision
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        work, precision = children
        return cls(work=work, precision=precision)


@jax.tree_util.register_pytree_node_class
class Timings:
    """Work-precision diagram timings."""

    def __init__(self, min, max, mean, stdev, data):  # noqa: A002
        """Construct a collection of timings."""
        self.min = min
        self.max = max
        self.mean = mean
        self.stdev = stdev
        self.data = jnp.asarray(data)

    @classmethod
    def from_list(cls, timings):
        return cls(
            min=min(timings),
            max=max(timings),
            mean=statistics.mean(timings),
            stdev=statistics.stdev(timings),
            data=jnp.asarray(timings),
        )

    def tree_flatten(self):
        children = self.min, self.max, self.mean, self.stdev, self.data
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        min, max, mean, stdev, data = children  # noqa: A001
        return cls(min=min, max=max, mean=mean, stdev=stdev, data=data)


# Work-precision diagram functionality


def create(problem, methods):
    return {
        method.label: (
            _evaluate_method(method_config=method, problem_config=problem),
            method,
        )
        for method in tqdm(methods)
    }


def _evaluate_method(*, method_config, problem_config):
    method = method_config.method
    if method_config.key is not None:
        problem = problem_config.problems[method_config.key]
        solve_fn = problem_config.solve_fns[method_config.key]
    else:
        problem = problem_config.problems
        solve_fn = problem_config.solve_fns

    error_fn = problem_config.error_fn
    atols = problem_config.atols
    rtols = problem_config.rtols
    repeat = problem_config.repeat

    def fn(atol, rtol):
        return solve_fn(*problem.args, **problem.kwargs, atol=atol, rtol=rtol, **method)

    if method_config.jit:
        if method_config.tols_static:
            fn = jax.jit(fn, static_argnames=("atol", "rtol"))
        else:
            fn = jax.jit(fn)

    results = [
        _evaluate_method_and_tolerance(
            error_fn=error_fn, fn=fn, rtol=rtol, atol=atol, repeat=repeat
        )
        for atol, rtol in zip(tqdm(atols, leave=False), rtols)
    ]
    return tree_array_util.tree_stack(results)


def _evaluate_method_and_tolerance(*, error_fn, fn, atol, rtol, repeat):
    qoi = fn(atol=float(atol), rtol=float(rtol))
    error = error_fn(qoi)
    timings = timeit.repeat(
        lambda: fn(atol=float(atol), rtol=float(rtol)), number=1, repeat=repeat
    )
    return Results(work=Timings.from_list(timings), precision=error)


def plot(
    *,
    results: Dict[Any, Tuple[Results, MethodConfig]],
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
    for solver, (result, method) in results.items():
        ax.loglog(
            result.precision,
            result.work.mean,
            label=solver,
            alpha=alpha_mean,
            **method.plotting_kwargs,
        )
        ax.fill_between(
            result.precision,
            result.work.mean - result.work.stdev,
            result.work.mean + result.work.stdev,
            alpha=alpha_stdev,
            **method.plotting_kwargs,
        )

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_xticks(yticks)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None and ylabel_unit is not None:
        xlabel += f" [{xlabel_unit}]"
        ax.set_xlabel(xlabel)
    if ylabel is not None and ylabel_unit is not None:
        ylabel += f" [{ylabel_unit}]"
        ax.set_ylabel(ylabel)

    ax.grid(which_grid)
    ax.legend()
    return fig, ax
