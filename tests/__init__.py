"""Tests."""
import os

from tests.setup import setup

if "IMPL" not in os.environ:
    msg = "Select an implementation"
    raise KeyError(msg)

setup.select(os.environ["IMPL"])
