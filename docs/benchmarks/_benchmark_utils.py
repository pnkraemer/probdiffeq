"""Benchmark utils."""

import subprocess
import timeit

import numpy as np
from tueplots import axes, bundles, cycler, markers
from tueplots.constants import markers as marker_constants
from tueplots.constants.color import palettes


def plot_config():
    return {
        **bundles.beamer_moml(rel_width=1.0, rel_height=1.0),
        **axes.color(base="black"),
        **axes.grid(),
        **cycler.cycler(
            marker=np.tile(marker_constants.o_sized[:9], 4)[:15],
            color=np.tile(palettes.muted, 4)[:15],
        ),
        **markers.with_edge(),
        "figure.dpi": 150,
    }


def most_recent_commit(*, abbrev=21):
    return subprocess.check_output(
        ["git", "describe", "--always", f"--abbrev={abbrev}"]
    )


def time(fn, /, *, number=10, repeat=10):
    res = fn()
    t = min(timeit.repeat(fn, number=number, repeat=repeat)) / number
    return t, res
