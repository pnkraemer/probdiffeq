"""Benchmark utils."""

import subprocess
import timeit

from tueplots import axes, bundles, cycler, markers
from tueplots.constants import markers as marker_constants
from tueplots.constants.color import palettes


def plot_config():
    return {
        **bundles.beamer_moml(),
        **axes.color(base="black"),
        **axes.grid(),
        **cycler.cycler(marker=marker_constants.o_sized[:5], color=palettes.muted[:5]),
        **markers.with_edge(),
    }


def most_recent_commit(*, abbrev=21):
    return subprocess.check_output(
        ["git", "describe", "--always", f"--abbrev={abbrev}"]
    )


def time(fn, /, *, number=10, repeat=1):
    res = fn()
    t = min(timeit.repeat(fn, number=number, repeat=repeat)) / repeat
    return t, res
