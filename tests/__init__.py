"""Tests."""
import os

from tests.setup import setup

if "IMPL" not in os.environ:
    raise KeyError("Select an implementation")

setup.select(os.environ["IMPL"])
