"""System information (most recent commit, JAX version, etc.)."""

import subprocess

import jax

import probdiffeq


def print_info():
    commit = _most_recent_commit(abbrev=6)

    print(f"probdiffeq version:\n\t{probdiffeq.__version__}")

    # todo: the probdiffeq version should suffice now, right?
    print(f"Most recent commit:\n\t{commit}")
    print()
    jax.print_environment_info()


def _most_recent_commit(*, abbrev=21):
    return subprocess.check_output(
        ["git", "describe", "--always", f"--abbrev={abbrev}"]
    )
