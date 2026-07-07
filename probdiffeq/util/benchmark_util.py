"""Shared utilities for benchmark scripts."""

from probdiffeq.backend import linalg, np, structs, timing, tree
from probdiffeq.backend.typing import Array, Callable

# TODO: this being a function in the src is a joke...


def setup_tolerances(*, start: float, stop: float, step: float) -> Array:
    """Choose a vector of tolerances."""
    return 0.1 ** np.arange(start, stop, step=step)


@tree.register_dataclass
@structs.dataclass
class WorkPrec:
    """Data for work-precision diagrams."""

    arg: Array
    """The inputs for the work-precision diagrams."""

    work_first_run: Array
    """The time it took for the initial run (includes compilation time)."""

    work: Array
    """A set of runtimes."""

    precision: Array
    """The precision respectively target functions."""


def workprec(
    target_fun: Callable,
    /,
    *,
    num_timing_calls: int = 1,
    time_max_sec=30,
    print_progress: bool = False,
) -> Callable[[list], WorkPrec]:
    """Turn a parameter-to-precision function into a parameter-to-work-precision map."""

    def parameter_list_to_workprecision(list_of_args: list, /) -> WorkPrec:
        workprecs = []
        time_start = timing.perf_counter()

        for arg in list_of_args:
            time_elapsed = timing.perf_counter() - time_start
            if time_elapsed > time_max_sec:
                break

            if print_progress:
                msg = f"arg = {arg} | elapsed: {time_elapsed:.2f} | time_max_sec = {time_max_sec}"
                print(msg)  # noqa: T201

            # Compile...
            t0 = timing.perf_counter()
            precision = target_fun(arg)
            precision = tree.tree_map(lambda x: x.block_until_ready(), precision)
            work_first_run = np.asarray(timing.perf_counter() - t0)

            # Call many times to get a set of runtimes
            works = []
            for _ in range(num_timing_calls):
                t0 = timing.perf_counter()
                y = target_fun(arg)
                _ = tree.tree_map(lambda x: x.block_until_ready(), y)
                work = np.asarray(timing.perf_counter() - t0)
                works.append(work)

            arg = np.asarray(arg)
            workprec = WorkPrec(
                arg=arg,
                work_first_run=work_first_run,
                work=np.asarray(works),
                precision=precision,
            )
            workprecs.append(workprec)

        # Transpose workprec tree again to return a WorkPrec object
        return tree.tree_array_stack(workprecs)

    return parameter_list_to_workprecision


def rmse_relative(expected: Array, *, nugget=1e-5) -> Callable:
    """Compute the relative RMSE."""
    expected = np.asarray(expected)

    def rmse(received):
        received = np.asarray(received)
        error_absolute = np.abs(expected - received)
        error_relative = error_absolute / np.abs(nugget + expected)
        return linalg.vector_norm(error_relative) / np.sqrt(error_relative.size)

    return rmse


def rmse_absolute(expected: Array) -> Callable:
    """Compute the absolute RMSE."""
    expected = np.asarray(expected)

    def rmse(received):
        received = np.asarray(received)
        error_absolute = np.abs(expected - received)
        return linalg.vector_norm(error_absolute) / np.sqrt(error_absolute.size)

    return rmse


def adaptive_repeat(xs, ys):
    """Repeat doubling values to create a comprehensible plot."""
    zs = []
    for x, y in zip(xs, ys):
        zs.extend([x] * int(y))
    return np.asarray(zs)
