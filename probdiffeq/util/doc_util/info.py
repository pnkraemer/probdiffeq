"""System information (most recent commit, JAX version, etc.)."""

import subprocess

import diffrax
import jax
import scipy

import probdiffeq


def print_info():
    print()
    print(f"ProbDiffEq version:\n\t{probdiffeq.__version__}")
    print(f"Diffrax version:\n\t{diffrax.__version__}")
    print(f"SciPy version:\n\t{scipy.__version__}")
    print()
    # TODO: the probdiffeq version should suffice now, right?
    commit = _most_recent_commit(abbrev=6)
    print(f"Most recent ProbDiffEq commit:\n\t{commit}")
    print()
    jax.print_environment_info()


def _most_recent_commit(*, abbrev=21):
    return subprocess.check_output(
        ["git", "describe", "--always", f"--abbrev={abbrev}"]
    )
