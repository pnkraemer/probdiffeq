"""Shared utilities for benchmark scripts."""

from probdiffeq.backend import linalg, np, structs, timing, tree
from probdiffeq.backend.typing import Array, Callable


def setup_tolerances(*, start: float, stop: float, step: float) -> Array:
    """Choose a vector of tolerances."""
    return 0.1 ** np.arange(start, stop, step=step)


@tree.register_dataclass
@structs.dataclass
class WorkPrec:
    """Data for work-precision diagrams."""

    work: Array
    precision: Array


def workprec(
    precision_fun: Callable, *, num_timing_calls: int = 1
) -> Callable[[list], WorkPrec]:
    """Turn a parameter-to-precision function into a parameter-to-work-precision map."""

    def parameter_list_to_workprecision(list_of_args: list, /) -> WorkPrec:
        workprecs = []
        for arg in list_of_args:
            workprec = []

            # Compile...
            y = precision_fun(arg)
            tree.tree_map(lambda x: x.block_until_ready(), y)

            # Loop
            for _ in range(num_timing_calls):
                t0 = timing.perf_counter()
                precision = precision_fun(arg)
                tree.tree_map(lambda x: x.block_until_ready(), precision)
                work = np.asarray(timing.perf_counter() - t0)
                workprec.append(WorkPrec(work=work, precision=precision))

            # Transpose workprec tree
            workprec = tree.tree_array_stack(workprec)
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


def adaptive_benchmark(
    fun, *, timeit_fun: Callable, max_time, print_progress: bool = True
) -> dict:
    """Benchmark a function iteratively until a max-time threshold is exceeded."""
    work_compile = []
    work_mean = []
    work_std = []
    arguments = []

    t0 = timing.perf_counter()
    arg = 1
    while (elapsed := timing.perf_counter() - t0) < max_time:
        if print_progress:
            msg = f"num = {arg} | elapsed = {elapsed:.2f} | max_time = {max_time}"
            print(msg)  # noqa: T201
        t0 = timing.perf_counter()
        tcoeffs = fun(arg).block_until_ready()
        t1 = timing.perf_counter()
        time_compile = t1 - t0

        time_execute = timeit_fun(lambda: fun(arg).block_until_ready())  # noqa: B023

        arguments.append(len(tcoeffs))
        work_compile.append(time_compile)
        work_mean.append(np.mean(np.asarray(time_execute)))
        work_std.append(np.std(np.asarray(time_execute), ddof=1))
        arg += 1

    if print_progress:
        msg = f"num = {arg} | elapsed = {elapsed:.2f} | max_time = {max_time}"
        print(msg)  # noqa: T201
    return {
        "work_mean": np.asarray(work_mean),
        "work_std": np.asarray(work_std),
        "work_compile": np.asarray(work_compile),
        "arguments": np.asarray(arguments),
    }


def adaptive_repeat(xs, ys):
    """Repeat doubling values to create a comprehensible plot."""
    zs = []
    for x, y in zip(xs, ys):
        zs.extend([x] * int(y))
    return np.asarray(zs)
