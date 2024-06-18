"""Test-setup."""

import os

from probdiffeq.backend import config, containers, ode, testing, warnings
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any
from probdiffeq.impl import impl

# All warnings shall be errors
warnings.filterwarnings("error")

# Test on CPU.
config.update("platform_name", "cpu")

# Double precision
# Needed for equivalence tests for smoothers.
config.update("enable_x64", True)


@containers.dataclass
class SSMConfig:
    """State-space model configuration."""

    default_rv: Any
    default_ode: Any
    default_ode_affine: Any


# todo: if we replace the fixture with cases, we can opt-in the tests
#  to running with different configurations instead of opt-out.
@testing.fixture(name="ssm")
def fixture_ssm():
    """Select a state-space model factorisation."""
    if "IMPL" not in os.environ:
        msg = "Select an implementation"
        raise KeyError(msg)

    which = os.environ["IMPL"]
    if which in ["dense", "isotropic", "blockdiag"]:
        impl.select(which, ode_shape=(2,))

        # Prepare an RV:
        output_scale = np.ones_like(impl.prototypes.output_scale())
        discretise_func = impl.conditional.ibm_transitions(3, output_scale)
        (_matrix, rv), _pre = discretise_func(0.5)

        # Prepare an ODE
        ode_ = ode.ivp_lotka_volterra()
        ode_affine = ode.ivp_affine_multi_dimensional()

        # Return the SSM config
        return SSMConfig(default_rv=rv, default_ode=ode_, default_ode_affine=ode_affine)
    if which in ["scalar"]:
        impl.select("scalar")

        # Prepare an RV
        output_scale = np.ones_like(impl.prototypes.output_scale())
        discretise_func = impl.conditional.ibm_transitions(3, output_scale)
        (_matrix, rv), _pre = discretise_func(0.5)

        # Prepare an ODE
        ode_ = ode.ivp_logistic()
        ode_affine = ode.ivp_affine_scalar()

        # Return the SSM config
        return SSMConfig(default_rv=rv, default_ode=ode_, default_ode_affine=ode_affine)
    raise ValueError
