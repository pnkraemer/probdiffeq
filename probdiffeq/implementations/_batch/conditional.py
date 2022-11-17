from typing import Callable, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import cubature as cubature_module
from probdiffeq.implementations import _collections, _ibm_util, _scalar
