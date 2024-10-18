"""Test-setup."""

from probdiffeq.backend import config, warnings

# All warnings shall be errors
warnings.filterwarnings("error")

# Test on CPU.
config.update("platform_name", "cpu")

# Double precision
# Needed for equivalence tests for smoothers.
config.update("enable_x64", True)
